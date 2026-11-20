# Patches for on-chip-training

Changes the `dpd` stage needs on the Pi's side, written here and applied by the user (nothing in
on-chip-training is changed from this repository). Each applies to branch `dpd` at commit `14a72ff`.

## 0001-co_optics-kappa-from-dpd_bb.patch

`testbed/rpi/src/stages/link_stages.{h,cpp}`, `CoOptics` (docs/testbed/jetson/STAGE.md, decisions 1a
and 3b):

1. **κ from the stage before it.** `configure` looks in the CONFIG frame's `pipeline` list for an
   earlier line with `calibration.kappa` (the `dpd` stage's, taken from the undistorted block), as
   `fiber` finds `field_norm`; at the block's end `co_optics` uses that κ instead of its own. With
   the DPD in bypass no line carries one, and `co_optics` computes κ as before.
2. **The block by its samples' indices.** At `CALIB_END` the block is the samples from
   `block.first_symbol × sps` on (as `ue_frontend` and `ue_dsp` already locate it), not the first
   `n_block` samples of the group: the `dpd` stage sends its group from 250 samples earlier (its
   holdback). With no DPD the group starts at the block, so the same samples are taken in the same
   order: κ and `field_norm` are bit for bit as before.

Not compiled here (the Pi's stages are not built on the Jetson); `git apply --check` passes against
`14a72ff`. To apply and check, on the Pi:

```bash
cd ~/on-chip-training
git apply ~/<path>/0001-co_optics-kappa-from-dpd_bb.patch
make -C testbed/rpi                                   # rebuild the image
# the pipeline is still bit for bit the single program with both DPD places in bypass:
#   ctest (test_pipeline_bitexact) in the build, as docs/testbed/rpi/PIPELINE.md, Section 8
```

## compose.jetson.yaml

Not a patch but a new file for `testbed/rpi/`, beside `compose.yaml`, which stays unchanged: the
override that puts the Jetson's `dpd` stage in `dpd_bb`'s place (`source` sends to the Jetson,
`dpd_bb` not started, `co_optics` published on port 7002, `pa`'s tap to the Jetson). Checked with
`docker compose config` on copies of the two files. Its use and the order of the bring-up:
`docs/testbed/jetson/HOOKUP.md`.
