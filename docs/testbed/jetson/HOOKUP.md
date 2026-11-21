# The hookup: the Jetson's `dpd` stage in the Pi's pipeline

Status: **proposed** (2026-11-20). Nothing on the Pi has been changed or run from this repository;
every step on the Pi is the user's. The Jetson's side is ready (`docker/compose.yaml`).

## The wiring

Both machines on the testbed's wired subnet 192.168.91.0/25; the KV260 is not involved.

```
 Pi 4 (rpi4b, 192.168.91.80)                         Jetson (orinnano8s, 192.168.91.89)
 source ── AROF_OUT tcp://192.168.91.89:7000 ──────► dpd_bb (dpd_stage), listens on 7000
 co_optics, port 7000 published as 7002 ◄─────────── AROF_OUT tcp://192.168.91.80:7002
   └─ fiber ─ pd_rau ─ dpd_rf ─ pa ─ wireless ─ ue_frontend ─ ue_dsp ─ streamer ─► KV260
                               pa  ─ AROF_TAP tcp://192.168.91.89:7107 ────────► tap_pa (recorder), y
                                                     dpd_bb ─ its own tap ─────► tap_dpd (recorder), x_d
```

| Where | Change | File |
|---|---|---|
| Pi | `source` sends to the Jetson; `dpd_bb` not started (the Jetson takes its place); `co_optics` publishes port 7002; `pa`'s tap goes to the Jetson | `patches/on-chip-training/compose.jetson.yaml`, an override used next to the unchanged `compose.yaml` |
| Pi | `co_optics` takes κ from the `dpd` stage and locates the calibration block by `first_sample` | `patches/on-chip-training/0001-co_optics-kappa-from-dpd_bb.patch` (needed for dpd mode, harmless in bypass) |
| Pi | the settings file gains `"dpd"`, and `"taps"` for a recording (below) | a copy of `config/testbed_rpi/nominal.json` |
| Jetson | the stage and the two recorders | `docker/compose.yaml` (ready) |

Checked here: the override merges as intended (`docker compose config` on copies of the files: no
`dpd_bb`, `source` and `pa`'s tap to 192.168.91.89, `co_optics` published on 7002); the patch applies
to on-chip-training `14a72ff` (`git apply --check`); the Jetson has no firewall (ufw inactive,
iptables accepting).

Read on the Pi over SSH (2026-11-21, `ssh rpi4b`, read-only): no firewall (nftables, iptables, ufw
inactive); nothing running; the image `arof-stream:latest`; `~/on-chip-training` on branch
`paper-rewrite` at `b6f4e49`, an ancestor of `14a72ff` with `testbed/rpi` identical: the stage
library copied here is the code the Pi runs, and the patch and the override apply to its clone as
they are. SSH works both ways (`ssh rpi4b` from the Jetson, `ssh orinnano8s` from the Pi).

## The settings

The source carries the settings file's `"dpd"`, `"taps"` and `"log"` sections to every stage in the
CONFIG frame (on-chip-training `source_runner.cpp`). For the runs below, in a copy of
`nominal.json` (the `//` comments are this page's, not JSON):

```json
"dpd":  {"bb": {"mode": "bypass"}},                       // or "dpd" with "model": "<weights>"
"taps": {"pa": "lossless", "dpd_bb": "lossless"},         // for a recording; omit otherwise
"log":  {"timing": true}                                  // per-stage CSVs: the Pi's in logs/, dpd_bb.csv on the Jetson
```

## The clocks

Set up on 2026-11-21: chrony on both, the Pi serving the subnet, the Jetson following the Pi only;
offset between the two ~2 µs (standard deviation). What was changed on each machine, the checks,
how to undo it, and the KV260 for later: [CLOCKS.md](CLOCKS.md).

## Bring-up, in order

Each step needs the one before it to pass.

1. **The Jetson's stage, alone** (done): `make -C docker check` (the GPU in the container) and
   `DPD_STAGE_DOCKER=1 python -m pytest tests/test_dpd_stage.py` (the stage in the image between a
   feeder and a sink).
2. **Bypass through the Jetson**, settings `"dpd": {"bb": {"mode": "bypass"}}`, the override in use
   (the patch is not needed yet). The Jetson forwards every frame byte for byte, so the stream to the
   KV260 must be the single program's, bit for bit, as with `dpd_bb` on the Pi (PIPELINE.md,
   Section 8, test 3; `check_board_run.py`). Start the Jetson first (it waits up to 10 min for the
   Pi), then the Pi:
   ```bash
   # Jetson
   docker compose -f docker/compose.yaml up
   # Pi, ~/on-chip-training/testbed/rpi
   docker compose -f compose.yaml -f compose.jetson.yaml up --abort-on-container-exit --exit-code-from streamer
   ```
   With `"log": {"timing": true}` this run also gives the first latency of the round trip
   Pi → Jetson → Pi (`dpd_bb.csv` against the Pi's `source.csv` and `co_optics.csv`), once the clocks
   are synchronized.
3. **Recording for ILA**: bypass, `"taps": {"pa": "lossless", "dpd_bb": "lossless"}`, the Jetson
   with its recorders (`docker compose -f docker/compose.yaml --profile tap up`). The Jetson gets
   `records/dpd_tap.bin` (the drive sent, x_d) and `records/pa_tap.bin` (the PA's output, y), frame
   by frame, matched by `seq`. Lossless taps slow the chain to their consumers' pace rather than drop.
4. **Training on the recordings** (the reference, in Python; not built yet): the PA's output aligned
   with the drive and the pilot tone removed (the DPD plan's step 3, `py/ila.py`), ILA in units of
   the drive in volts (`κ·s`, κ from the stage's `pipeline` line), the model exported
   (`scripts/export_model.py`).
5. **DPD on the link**: the patch applied on the Pi (and its image rebuilt; the bit-exact test in
   bypass passes as before), settings `"dpd": {"bb": {"mode": "dpd", "model": "<the trained weights>"}}`;
   the PA's output and the board's EVM and BER against step 2.

## Failure and stop

As the contract has it (PIPELINE.md 3.5): if the Jetson's stage exits, the source's connection
fails and the Pi's pipeline stops (`--abort-on-container-exit`); if a Pi stage exits, `co_optics`
closes its connection and the Jetson's stage stops with an error. The END frame stops both in order:
the Jetson's stage forwards it, waits for `co_optics` to close, and exits 0. Neither side restarts on
its own.
