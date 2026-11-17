# How to run it

| Page | Device |
|---|---|
| [jetson.md](jetson.md) | the Jetson Orin Nano: the conda env, the tests, training, the stage's image |
| [pc.md](pc.md) | the PC: the reference DPD on the CPU, called from MATLAB |

Every device holds a clone of the same repository (`~/dpd`): change code, push, `git pull` on the
other device. The Pi's side (the pipeline) is in on-chip-training's `docs/howto/rpi.md`; the
live run with the Jetson in the pipeline comes with step 3.
