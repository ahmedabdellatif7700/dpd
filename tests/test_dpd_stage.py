"""The dpd stage (cpp/bin/dpd_stage) between a feeder and a sink that stand in for the Pi: the feeder
sends what the source sends (a CONFIG frame, the calibration block in 128-symbol frames with their
REF frames, the tail, data frames, END), the sink records what comes out, the stage runs between them
over TCP as in the pipeline. Skipped when the stage is not built.

  bypass: every frame leaves byte for byte as it came (the CONFIG frame gains the stage's line in
          its "pipeline" list, as the contract has every stage do);
  dpd:    kappa in the stage's line equals co_optics's formula on the undistorted block; REF passes
          unchanged; seq, flags and order are kept; the BASEBAND equals the reference applied offline
          (the PyTorch model and the resampler through dpd/stream.py), on CUDA and on the CPU;
  errors: a stage before it that does not produce BASEBAND stops it, with a message naming both.

DPD_STAGE_DOCKER=1 runs the stage in the image instead (arof-dpd:local, docker run --runtime nvidia
--network host, the models mounted at /models): the same tests, against what runs in the pipeline.
"""
import os
import socket
import subprocess
import threading
import time

import numpy as np
import pytest

from dpd import frame2 as F
from dpd import resample as R
from dpd import signals
from dpd.models import DPD
from dpd.stream import FrameStream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "cpp", "bin", "dpd_stage")
MODELS = os.path.join(ROOT, "results", "models")
MODEL = "transformer_nominal"
SPS, CHUNK = 50, 128                         # the source's: 50 samples per symbol, 128-symbol frames
BLOCK, TAIL, DATA = 4 * CHUNK + 40, 30, 3    # block symbols (not a multiple of the chunk), tail, data frames

DOCKER = os.environ.get("DPD_STAGE_DOCKER") == "1"
pytestmark = pytest.mark.skipif(not DOCKER and not os.path.exists(BIN), reason="cpp/bin/dpd_stage not built (make -C cpp)")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def source_frames(mode, dtype=np.complex128, upstream_out="BASEBAND", taps=None):
    """what the Pi's source sends (Source::calibration_block, emit): returns the frames"""
    n_sym = BLOCK + TAIL + DATA * CHUNK
    s, d = signals.baseband(n_sym, sps=SPS, seed=11)
    labels = np.arange(n_sym) % 16
    settings = {"settings": {"tx": {"fs": 20e9, "rs": 0.4e9}, "link": {"vpp_mv": 500.0}, "stages": {}},
                "block": {"first_symbol": 0, "block_symbols": BLOCK, "tail_symbols": TAIL, "guard": 16},
                "pipeline": [{"stage": "source", "out": upstream_out, "dtype": "c128", "fs": 20e9}],
                "dpd": {"bb": {"mode": mode, "model": MODEL}}, "log": {"timing": True}}
    if taps:
        settings["taps"] = taps
    frames, seq = [F.config(settings, seq=0, capture_time_ns=1000)], 1

    def emit(i0, n, flags):
        nonlocal seq
        ref = np.stack([d[i0:i0 + n], labels[i0:i0 + n].astype(complex)], 1).ravel()
        frames.append(F.wave(F.REF, ref, i0, seq, flags, words_per_sample=2, capture_time_ns=1000 + seq))
        frames.append(F.wave(F.BASEBAND, s[i0 * SPS:(i0 + n) * SPS].astype(dtype), i0 * SPS, seq + 1, flags,
                             capture_time_ns=1000 + seq + 1))
        seq += 2

    for i in range(0, BLOCK, CHUNK):
        emit(i, min(CHUNK, BLOCK - i), F.CALIB)
    emit(BLOCK, TAIL, F.CALIB | F.CALIB_END)
    for k in range(DATA):
        emit(BLOCK + TAIL + k * CHUNK, CHUNK, 0)
    frames.append(F.end(seq, capture_time_ns=1000 + seq))
    return frames, s


def run_stage(frames, tmp_path, device="cuda", tap=None):
    """the stage between the feeder and the sink; returns (frames out, exit code, stderr). tap: an
    endpoint for the stage's AROF_TAP"""
    p_in, p_out = free_port(), free_port()
    sink_sock = socket.socket()
    sink_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sink_sock.bind(("127.0.0.1", p_out))
    sink_sock.listen(1)
    sink_sock.settimeout(60)                     # a test that goes wrong fails, it does not hang
    got = []

    def sink():                                  # co_optics's place: records, closes after END
        try:
            c, _ = sink_sock.accept()
        except OSError:
            return
        with c:
            c.settimeout(120)
            while (f := F.read_frame(c)) is not None:
                got.append(f)
                if f.flags & F.END:
                    break

    t = threading.Thread(target=sink, daemon=True)
    t.start()
    arof = {"AROF_STAGE": "dpd_bb", "AROF_IN": f"tcp://127.0.0.1:{p_in}", "AROF_OUT": f"tcp://127.0.0.1:{p_out}",
            "AROF_DPD_DEVICE": device, "AROF_CONNECT_TIMEOUT_S": "20"}
    if tap:
        arof["AROF_TAP"] = tap
    if DOCKER:                                   # the stage in its image, on the host's network
        name = f"dpd_stage_test_{p_in}"
        cmd = ["docker", "run", "--rm", "--name", name, "--runtime", "nvidia", "--network", "host",
               "--user", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{MODELS}:/models:ro", "-v", f"{tmp_path}:/logs"]
        for k, v in {**arof, "AROF_MODELS_DIR": "/models", "AROF_LOG_DIR": "/logs"}.items():
            cmd += ["-e", f"{k}={v}"]
        proc = subprocess.Popen(cmd + ["arof-dpd:local", "dpd_stage"], stderr=subprocess.PIPE, text=True)
    else:
        name = None
        env = dict(os.environ, **arof, AROF_MODELS_DIR=MODELS, AROF_LOG_DIR=str(tmp_path))
        proc = subprocess.Popen([BIN], env=env, stderr=subprocess.PIPE, text=True)
    feed = None
    for _ in range(600):                         # the source's place: connect when the stage listens
        try:
            feed = socket.create_connection(("127.0.0.1", p_in), timeout=5)
            break
        except OSError:
            time.sleep(0.05)
    try:
        assert feed is not None, "the stage does not listen"
        try:
            with feed:
                for f in frames:
                    F.write_frame(feed, f)
        except (BrokenPipeError, ConnectionResetError):
            pass                                 # the stage stopped early; its exit code and message say why
        _, err = proc.communicate(timeout=300)
    except BaseException:
        proc.kill()
        if name:
            subprocess.run(["docker", "kill", name], capture_output=True)
        _, err = proc.communicate()
        raise AssertionError(f"the stage run failed; its stderr:\n{err}")
    finally:
        t.join(timeout=30)
        sink_sock.close()
    return got, proc.returncode, err


def test_bypass_is_byte_exact(tmp_path):
    frames, _ = source_frames("bypass")
    got, code, err = run_stage(frames, tmp_path)
    assert code == 0, err
    assert len(got) == len(frames)
    for a, b in zip(frames, got):
        if a.flags & F.CONFIG:                   # the CONFIG frame: the stage's line appended
            ja, jb = a.values(), b.values()
            assert jb["pipeline"][:-1] == ja["pipeline"]
            assert jb["pipeline"][-1] == {"stage": "dpd_bb", "in": "BASEBAND", "out": "BASEBAND", "mode": "bypass"}
            jb["pipeline"] = ja["pipeline"]
            assert jb == ja and (b.seq, b.flags, b.config_id) == (a.seq, a.flags, a.config_id)
        else:
            assert b.to_bytes() == a.to_bytes()  # byte for byte
    log = (tmp_path / "dpd_bb.csv").read_text().splitlines()
    assert log[0].startswith("seq,signal,flags") and len(log) == 1 + len(frames)


def reference(frames, s):
    """the dpd stage offline: kappa from the block (co_optics's formula), then the stream of
    DPD(kappa s)/kappa, the group flushed whole"""
    block = s[:BLOCK * SPS]
    kappa = 1.0 * 0.5 / (block.real.max() - block.real.min())
    dpd, h = DPD.load(os.path.join(MODELS, MODEL + ".pt"), "cpu"), R.lowpass()

    def transform(v):
        return R.to_pipeline(dpd.predistort(R.to_dpd(kappa * v, h)), h)[:v.size] / kappa

    st, out = FrameStream(transform), np.zeros(0, complex)
    bb = [f for f in frames if f.signal == F.BASEBAND]
    for i, f in enumerate(bb):
        x = f.values().astype(np.complex64).astype(complex)  # the stage takes float32 values
        _, y = st.push(x)
        out = np.concatenate([out, y])
        if f.flags & F.CALIB_END:
            _, y = st.flush()
            out = np.concatenate([out, y])
    return kappa, out


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_dpd_mode(tmp_path, device):
    frames, s = source_frames("dpd")
    got, code, err = run_stage(frames, tmp_path, device)
    assert code == 0, err
    # one frame out per frame in, in order, seq and flags kept
    assert len(got) == len(frames)
    assert [(f.seq, f.flags) for f in got] == [(f.seq, f.flags) for f in frames]
    # kappa: in the stage's line, equal to co_optics's formula on the undistorted block
    kappa_ref, y_ref = reference(frames, s)
    entry = got[0].values()["pipeline"][-1]
    assert entry["stage"] == "dpd_bb" and entry["mode"] == "dpd" and entry["model"] == MODEL
    assert entry["calibration"]["kappa"] == kappa_ref
    # REF and END pass unchanged
    for a, b in zip(frames, got):
        if a.signal == F.REF or a.flags & F.END:
            assert b.to_bytes() == a.to_bytes()
    # BASEBAND: contiguous samples, numbered by first_sample, equal to the reference
    bb = [f for f in got if f.signal == F.BASEBAND]
    pos = 0
    for f in bb:
        assert f.first_sample == pos and f.dtype == F.C128
        pos += f.n_samples
    y = np.concatenate([f.values() for f in bb])
    assert y.size == y_ref.size == DATA * CHUNK * SPS + (BLOCK + TAIL) * SPS - 250   # the holdback waits at END
    err_rel = np.max(np.abs(y - y_ref)) / np.sqrt(np.mean(np.abs(y_ref) ** 2))
    assert err_rel < 1e-5, err_rel
    assert np.max(np.abs(y - np.concatenate([f.values() for f in frames if f.signal == F.BASEBAND])[:y.size])) > 1e-3
    # the calibration group leaves whole: the tail's frame ends where the group does
    tail = [f for f in got if f.signal == F.BASEBAND and f.flags & F.CALIB_END][0]
    assert tail.first_sample + tail.n_samples == (BLOCK + TAIL) * SPS


def test_wrong_pipeline_stops_it(tmp_path):
    frames, _ = source_frames("dpd", upstream_out="OPTICAL")
    _, code, err = run_stage(frames[:1], tmp_path)
    assert code == 1 and "consumes BASEBAND" in err and "OPTICAL" in err


RECORDER = os.path.join(ROOT, "cpp", "bin", "tap_recorder")


@pytest.mark.skipif(not os.path.exists(RECORDER), reason="cpp/bin/tap_recorder not built")
def test_tap_to_the_recorder(tmp_path):
    """the stage's own tap (the drive it sent, x_d for ILA), lossless, recorded by tap_recorder: the
    file holds the frames the stage sent on, BASEBAND, CONFIG and END, byte for byte"""
    p_tap = free_port()
    rec_file = tmp_path / "dpd_tap.bin"
    rec = subprocess.Popen([RECORDER], env=dict(os.environ, AROF_STAGE="tap_dpd", AROF_IN=f"tcp://127.0.0.1:{p_tap}",
                                                AROF_SINK_FILE=str(rec_file)), stderr=subprocess.PIPE, text=True)
    frames, _ = source_frames("dpd", taps={"dpd_bb": "lossless"})
    got, code, err = run_stage(frames, tmp_path, tap=f"tcp://127.0.0.1:{p_tap}")
    _, rec_err = rec.communicate(timeout=60)
    assert code == 0, err
    assert rec.returncode == 0 and "END" in rec_err, rec_err
    tapped = F.read_frames(rec_file.read_bytes())
    sent = [f for f in got if f.signal == F.BASEBAND or f.flags & (F.CONFIG | F.END)]
    assert len(tapped) == len(sent) == len(frames) - (len(frames) - 2) // 2   # all but the REF frames
    assert all(a.to_bytes() == b.to_bytes() for a, b in zip(tapped, sent))
