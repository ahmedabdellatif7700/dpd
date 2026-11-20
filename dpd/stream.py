"""The stage's processing of the BASEBAND stream (docs/testbed/jetson/STAGE.md, decision 3b): the
specification the C++ stage follows.

The resamplers and the model's window reach into the neighbouring frames: output sample n of
`to_pipeline(dpd(to_dpd(x)))` depends on the input from about n - 226 to n + 220 (at 50 samples per
symbol: each filter 62.75, the window 100 behind and 93.75 ahead). So the stream keeps HOLD samples
of each frame back until the next frame brings the look-ahead, and CONTEXT samples behind as the
past. Each push returns the samples it can complete, with their index: the signal is not delayed,
only packed differently, as the Pi's `fiber` stage does.

HOLD, CONTEXT and the frames are multiples of 25 samples (the rates' common grid: 25 samples at 50
per symbol are 4 at 8): every segment then starts on that grid, and the stream gives the drive's
whole transform, sample for sample, except within the reach of the stream's start (zeros before it)
and of a flush (zeros after it).
"""
import numpy as np

from . import resample as R

HOLD = 250       # samples kept for the look-ahead (>= 220)
CONTEXT = 250    # samples kept behind as the past (>= 226)
GRID = R.DOWN    # 25: the shared grid of the two rates, at 50 samples per symbol


class FrameStream:
    """transform: a segment at 50 samples per symbol -> the same length, taken alone (e.g. 50 -> 8,
    the model, 8 -> 50). push() takes the next frame, returns (first_sample, samples) completed."""

    def __init__(self, transform, hold=HOLD, context=CONTEXT):
        if hold % GRID or context % GRID:
            raise ValueError(f"hold and context must be multiples of {GRID}")
        self.transform, self.hold, self.context = transform, hold, context
        self.buf = np.zeros(0, dtype=np.complex128)   # samples from buf_start on
        self.buf_start = 0                            # index of buf[0] in the stream
        self.done = 0                                 # samples emitted so far (from the stream's start)

    def _emit(self, end):
        """transform what is held, return the samples from done to end"""
        y = self.transform(self.buf)[: self.buf.size]
        out = y[self.done - self.buf_start: end - self.buf_start]
        first, self.done = self.done, end
        keep_from = max(self.buf_start, self.done - self.context)
        self.buf, self.buf_start = self.buf[keep_from - self.buf_start:], keep_from
        return first, out

    def push(self, x):
        """the next frame (its size a multiple of 25); returns (first_sample, samples): all but the
        last `hold` samples held so far"""
        x = np.asarray(x)
        if x.size % GRID:
            raise ValueError(f"a frame must be a multiple of {GRID} samples")
        self.buf = np.concatenate([self.buf, x])
        end = self.buf_start + self.buf.size - self.hold
        if end <= self.done:
            return self.done, np.zeros(0, dtype=np.complex128)
        return self._emit(end)

    def flush(self):
        """everything held, its last samples with zeros after them (a calibration group leaves
        whole); the past is kept, since the stream goes on after the group"""
        return self._emit(self.buf_start + self.buf.size)
