# The hookup: the Jetson's `dpd` stage in the Pi's pipeline

Status: **in use** since 2026-11-21: the bypass run through the Jetson passed (below). The Pi's side
of the hookup is the `arof-pipeline` repository (`github.com/ahmedabdellatif7700/arof-pipeline`), a
copy of on-chip-training's pipeline (`14a72ff`) that runs on the Pi; on-chip-training itself is
unchanged. The Jetson's side is `docker/compose.yaml` here.

## The wiring

Both machines on the testbed's wired subnet 192.168.91.0/25; the KV260 is not involved.

```
 Pi 4 (rpi4b, 192.168.91.80)                         Jetson (orinnano8s, 192.168.91.89)
 source ── AROF_OUT tcp://192.168.91.89:7000 ──────► dpd_bb (dpd_stage), listens on 7000
 co_optics, port 7000 published as 7002 ◄─────────── AROF_OUT tcp://192.168.91.80:7002
   └─ fiber ─ pd_rau ─ dpd_rf ─ pa ─ wireless ─ ue_frontend ─ ue_dsp ─ streamer ─► receiver, port 5000
                               pa  ─ AROF_TAP tcp://192.168.91.89:7107 ────────► tap_pa (recorder), y
                                                     dpd_bb ─ its own tap ─────► tap_dpd (recorder), x_d
                                                ue_dsp ─ its tap ─► rx_tap (Pi): the received symbols
```

The receiver is the KV260, or without the board `tools/fake_kv260.py` on the Pi itself
(`stream.host` 192.168.91.80: the streamer runs in a container).

| Where | What | File |
|---|---|---|
| Pi | `source` sends to the Jetson; `dpd_bb` not started (the Jetson takes its place); `co_optics` publishes port 7002; `pa`'s tap goes to the Jetson | arof-pipeline `testbed/rpi/compose.jetson.yaml`, an override of the unchanged `compose.yaml`; `make pipeline … JETSON=1` |
| Pi | `co_optics` takes κ from the `dpd` stage and locates the calibration block by `first_sample` (decisions 1a and 3b, [STAGE.md](STAGE.md)) | arof-pipeline `testbed/rpi/src/stages/link_stages.{h,cpp}`; bit for bit as before with no DPD (`pipeline_bitexact` passes on the Pi, 2026-11-21) |
| Pi | the run's settings; `ue_dsp`'s tap and its recorder; the run's length | arof-pipeline `config/testbed_rpi/jetson_*.json`, `compose.yaml` (`rx_tap`), `SYMBOLS=` |
| Pi | the receiver's scores without an equalizer (EVM, SER, BER) | arof-pipeline `testbed/rpi/tools/evaluate_rx.py` |
| Jetson | the stage and the two recorders | `docker/compose.yaml` |

These were written first here as a patch and an override for on-chip-training (`patches/`, commit
`c26bfec`); they now live in arof-pipeline, which records each change from the copy (its README).

## The settings

The source carries the settings file's `"dpd"`, `"taps"` and `"log"` sections to every stage in the
CONFIG frame. Each run's file in arof-pipeline `config/testbed_rpi/` is `nominal.json` plus:

| Run | File | `"dpd"` | `"taps"` (lossless) |
|---|---|---|---|
| bypass | `jetson_bypass.json` | `{"bb": {"mode": "bypass"}}` | `ue_dsp` |
| recording | `jetson_record.json` | `{"bb": {"mode": "bypass"}}` | `dpd_bb`, `pa`, `ue_dsp` |
| DPD | (with the trained model) | `{"bb": {"mode": "dpd", "model": "<weights>"}}` | `pa`, `ue_dsp` |

All with `"log": {"timing": true}` (one CSV per stage: the Pi's in `testbed/rpi/logs/`, `dpd_bb.csv`
in `logs/` here) and `"stream": {"host": "192.168.91.80", "port": 5000}`.

## The clocks

Set up on 2026-11-21: chrony on both, the Pi serving the subnet, the Jetson following the Pi only;
offset between the two ~2 µs (standard deviation). What was changed on each machine, the checks,
how to undo it, and the KV260 for later: [CLOCKS.md](CLOCKS.md).

## Running

The Jetson first (its stage waits up to 10 min for the Pi), then the Pi, from
`~/arof-pipeline/testbed/rpi`:

```bash
# Jetson, ~/dpd
docker compose -f docker/compose.yaml [--profile tap] up         # the stage [and its recorders]
# Pi: the receiver, then the pipeline
python3 tools/fake_kv260.py --port 5000 --save records/stream.bin
make pipeline CONFIG=jetson_bypass JETSON=1 PROFILES=tap SYMBOLS=200000
python3 tools/evaluate_rx.py records/rx_tap.bin
```

`SYMBOLS=N` ends the run after N symbols in all (the calibration block's ~50 000 count): the source
sends END and every stage, the Jetson's included, exits 0 in order.

## Bring-up, in order

Each step needs the one before it to pass.

1. **The Jetson's stage, alone** (done, 2026-11-20): `make -C docker check` (the GPU in the
   container) and `DPD_STAGE_DOCKER=1 python -m pytest tests/test_dpd_stage.py` (the stage in the
   image between a feeder and a sink).
2. **Bypass through the Jetson** (done, 2026-11-21): `jetson_bypass`, 200 000 symbols. Results below.
3. **Recording for ILA** (done, 2026-11-21): `jetson_record`, the Jetson with its recorders
   (`--profile tap`). The Jetson gets `records/dpd_tap.bin` (the drive sent, x_d) and
   `records/pa_tap.bin` (the PA's output, y); the Pi gets `rx_tap.bin`. Lossless taps slow the
   chain to their consumers' pace rather than drop. Results below.
4. **Training on the recordings** (the reference, in Python; not built yet): the PA's output aligned
   with the drive and the pilot tone removed, ILA in units of the drive in volts (`κ·s`, κ from the
   stage's `pipeline` line), the model exported (`scripts/export_model.py`).
5. **DPD on the link**: settings with `"mode": "dpd"` and the trained weights; the PA's output and the
   receiver's EVM, SER and BER against step 2.

## Step 2: the bypass run (2026-11-21)

arof-pipeline `3c734b8` (image `arof-pipeline:dev`, `93c0242f160e`) on the Pi; `arof-dpd:local`
(`71cccc7a3ef4`) on the Jetson; `fake_kv260.py` on the Pi as the receiver. 16:08 to 16:15, ~7 min.

**The run.** Every container on both machines exited 0 at END: the Jetson's stage forwarded
3128 frames (2 CONFIG, 1563 REF, 1563 BASEBAND); every Pi stage logged 3128 frames. The receiver:
14 513 frames, one CONFIG, no RECALIBRATED, END received, no protocol errors; 5001.4 samples/s
against 5000 nominal (+0.03 %).

**The receiver without an equalizer** (`evaluate_rx.py`, `rx_tap.bin`, 16 constellation points):

| | Symbols | EVM | SER | BER |
|---|---|---|---|---|
| after the calibration | 149 866 | 19.46 % | 5.33e-2 | 1.33e-2 (7 999 bits) |
| the calibration block (`ue_dsp`'s own score) | | 19.54 % | | 1.33e-2 |

The DPD's baseline: the nominal link (500 mV, ROP −10 dBm, seed 2017) through the Jetson in bypass.
Not yet compared bit for bit with the single program's stream (`arof_stream`, the same settings),
which would show the bytes unchanged end to end; the stage forwarding byte for byte is checked by
`tests/test_dpd_stage.py`.

**Times** (the 1563 BASEBAND frames, from `dpd_bb.csv` and the Pi's `source.csv` and
`co_optics.csv`; the clocks within a few µs):

| | median | p99 | max |
|---|---|---|---|
| the stage, frame in to frame out | 108 µs | 208 ms | 2.46 s |
| of which processing (bypass: none) | 60 µs | 122 µs | 353 µs |
| source sent → stage received | 0.95 s | 11.0 s | 11.3 s |
| stage sent → `co_optics` received | 0.58 s | 10.5 s | 10.7 s |

All of these but the processing are back-pressure, not the Jetson or the network: the chain runs
at the stream's pace (5000 samples/s, set by the `streamer` at its end) while the source runs ahead,
so frames wait in the TCP buffers, and the stage's send (`t_send_ns` is after it) blocks while the
connection to `co_optics` is full: about one frame in 20 (133 of 3128 over 5 ms), up to 2.46 s.
The stage's own cost is the processing line.

## Step 3: the recording (2026-11-21)

The same versions as step 2; `jetson_record`, 200 000 symbols, 16:19:48 to 16:20:40 (52 s; step 2's
~7 min included `make pipeline` rebuilding the image after the pull, with its tests). Every
container on both machines exited 0 at END. Files in `records/2026-11-21_record/` (here, not in
git: 160 MB each) and `testbed/rpi/records/2026-11-21_record/` on the Pi, with the logs.

| Recording | Frames | Samples (index range, at 20 GS/s) | Gaps |
|---|---|---|---|
| `dpd_tap.bin`, x_d (`dpd_bb`'s output) | 1565 | 0 … 10 005 049 | none |
| `pa_tap.bin`, y (`pa`'s output) | 1565 | 0 … 9 999 231 | none |

Both start at the calibration block and run to END; y ends 5818 samples earlier (`fiber`'s
holdback, dropped at END). The two are paired by **sample index** (`first_sample`), not frame by
frame: `fiber` repacks the stream into 7936-sample frames, one in five or so empty, so frames of the
same `seq` hold different samples. κ for this run, from `co_optics`'s `pipeline` line:
1.2692 (`field_norm` 1.4903).

The receiver's scores are step 2's, to the digit (the same seed): 19.46 % EVM, 5.33e-2 SER,
1.33e-2 BER. y at the same index as x_d is not yet a usable pair: it carries the pilot tone (the
PA's input is the pilot at 1.104 a0 and the data at 0.371 a0 RMS), and a straight fit of y on x_d
leaves −0.4 dB; the delay between the two is a few samples (cross-correlation peak at 5, with the
pilot in). Removing the pilot and aligning are the training code's first steps (step 4).

## Failure and stop

As the contract has it (PIPELINE.md 3.5): if the Jetson's stage exits, the source's connection
fails and the Pi's pipeline stops (`--abort-on-container-exit`); if a Pi stage exits, `co_optics`
closes its connection and the Jetson's stage stops with an error. The END frame stops both in order:
the Jetson's stage forwards it, waits for `co_optics` to close, and exits 0 (seen in step 2).
Neither side restarts on its own.
