"""Frame version 2, the frame between the pipeline's stages, in Python: for the stage's tests (a
feeder and a sink standing in for the Pi) and for reading the PA tap's recordings.

The layout is on-chip-training's `testbed/rpi/src/stage/frame2.h` (PIPELINE.md, Section 2.3); the
header struct and the payload sizes as its reader `tap_frames()` (`testbed/rpi/tests/
test_pipeline_e2e.py`) unpacks them. Little-endian, a 44-byte header, then the payload:
n_samples * words_per_sample values of dtype, or n_samples bytes of JSON (CONFIG).
"""
import json
import struct
from dataclasses import dataclass, field

import numpy as np

HEADER = struct.Struct("<IHHIQIHBBIIQ")
MAGIC, VERSION = 0x46526F41, 2
# flags
RECALIBRATED, TEST, END, CONFIG, CALIB, CALIB_END = 1, 2, 4, 8, 16, 32
# dtype
Q8_24, C128, C64, JSON = 0, 1, 2, 3
DTYPE_BYTES = {Q8_24: 4, C128: 16, C64: 8, JSON: 1}
NUMPY = {C128: np.complex128, C64: np.complex64, Q8_24: np.int32}
# signal
NONE, REF, BASEBAND, OPTICAL, RF, RX_BASEBAND, RX_SYMBOLS, SAMPLES = range(8)


@dataclass
class Frame:
    flags: int = 0
    seq: int = 0
    first_sample: int = 0
    n_samples: int = 0
    words_per_sample: int = 0
    dtype: int = 0
    signal: int = NONE
    config_id: int = 0
    seed: int = 0
    capture_time_ns: int = 0
    payload: bytes = field(default=b"", repr=False)

    def to_bytes(self):
        return HEADER.pack(MAGIC, VERSION, self.flags, self.seq, self.first_sample, self.n_samples,
                           self.words_per_sample, self.dtype, self.signal, self.config_id, self.seed,
                           self.capture_time_ns) + self.payload

    def values(self):
        """the payload as a numpy array (complex for the waveforms), or the JSON (CONFIG)"""
        if self.flags & CONFIG:
            return json.loads(self.payload.decode())
        return np.frombuffer(self.payload, dtype=NUMPY[self.dtype])


def payload_bytes(flags, n_samples, words_per_sample, dtype):
    return n_samples if flags & CONFIG else n_samples * words_per_sample * DTYPE_BYTES[dtype]


def config(settings, seq=0, config_id=1, seed=0, capture_time_ns=0, flags=0):
    text = json.dumps(settings).encode()
    return Frame(flags=CONFIG | flags, seq=seq, n_samples=len(text), dtype=JSON, config_id=config_id, seed=seed,
                 capture_time_ns=capture_time_ns, payload=text)


def wave(signal, values, first_sample, seq, flags=0, words_per_sample=1, config_id=1, seed=0, capture_time_ns=0):
    """a data frame of complex values (complex128 or complex64, from the array's dtype)"""
    v = np.ascontiguousarray(values)
    dtype = C64 if v.dtype == np.complex64 else C128
    v = v.astype(NUMPY[dtype])
    return Frame(flags=flags, seq=seq, first_sample=first_sample, n_samples=v.size // words_per_sample,
                 words_per_sample=words_per_sample, dtype=dtype, signal=signal, config_id=config_id, seed=seed,
                 capture_time_ns=capture_time_ns, payload=v.tobytes())


def end(seq, config_id=1, seed=0, capture_time_ns=0):
    return Frame(flags=END, seq=seq, config_id=config_id, seed=seed, capture_time_ns=capture_time_ns)


def _read_exact(sock, n):
    b = bytearray()
    while len(b) < n:
        chunk = sock.recv(n - len(b))
        if not chunk:
            return None if not b else bytes(b)
        b += chunk
    return bytes(b)


def read_frame(sock):
    """the next frame from a socket, or None when it closes"""
    h = _read_exact(sock, HEADER.size)
    if h is None:
        return None
    if len(h) < HEADER.size:
        raise ValueError("frame2: a truncated header")
    magic, ver, flags, seq, first, n, wps, dtype, signal, cid, seed, t = HEADER.unpack(h)
    if magic != MAGIC or ver != VERSION:
        raise ValueError(f"frame2: bad header (magic {magic:#x}, version {ver})")
    size = payload_bytes(flags, n, wps, dtype)
    p = _read_exact(sock, size) if size else b""
    if size and (p is None or len(p) < size):
        raise ValueError("frame2: a truncated payload")
    return Frame(flags, seq, first, n, wps, dtype, signal, cid, seed, t, p)


def write_frame(sock, f):
    sock.sendall(f.to_bytes())


def read_frames(data):
    """the frames of a recording (a tap's file, bytes)"""
    out, i = [], 0
    while i + HEADER.size <= len(data):
        magic, ver, flags, seq, first, n, wps, dtype, signal, cid, seed, t = HEADER.unpack_from(data, i)
        if magic != MAGIC or ver != VERSION:
            raise ValueError(f"frame2: bad header at byte {i}")
        size = payload_bytes(flags, n, wps, dtype)
        out.append(Frame(flags, seq, first, n, wps, dtype, signal, cid, seed, t,
                         bytes(data[i + HEADER.size:i + HEADER.size + size])))
        i += HEADER.size + size
    if i != len(data):
        raise ValueError("frame2: trailing bytes")
    return out
