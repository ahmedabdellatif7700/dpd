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
iptables accepting). Not checked: anything on the Pi (its firewall, docker's port publishing there).

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

`capture_time_ns`, `t_recv_ns` and `t_send_ns` are compared across the two machines only as well as
their clocks agree. Today the Jetson follows the public pool with `systemd-timesyncd` (stratum 3,
19 ms root distance, polling every 34 min): far too coarse for latencies of a few milliseconds per
stage. Its network interface has no hardware time stamps (`ethtool -T`: software only), so PTP would
run in software, with no gain over NTP on this LAN.

**Proposed: chrony, the Pi serving time to the Jetson** over the wired LAN (typically tens of
microseconds of offset). System changes on both machines, for the user to make:

```bash
# Pi (Arch Linux ARM)
sudo pacman -S chrony
#   /etc/chrony.conf: keep its pool or server lines, add
#     allow 192.168.91.0/25
#     local stratum 10          # keeps serving the LAN if the internet is gone
sudo systemctl disable --now systemd-timesyncd 2>/dev/null; sudo systemctl enable --now chronyd

# Jetson (Ubuntu 24.04): installing chrony replaces systemd-timesyncd
sudo apt install chrony
#   /etc/chrony/chrony.conf: comment out the pool lines, add
#     server 192.168.91.80 iburst prefer minpoll 2 maxpoll 4
#     makestep 0.1 3
sudo systemctl restart chrony
chronyc tracking            # "System time" offset: should settle within ~0.1 ms
chronyc sources -v          # 192.168.91.80 marked '*'
```

The logs then carry the clocks' agreement with them: record `chronyc tracking` on the Jetson at the
start and the end of every run whose latencies are analysed.

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
