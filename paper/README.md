# The paper

A transformer DPD on an embedded GPU for a 28 GHz A-RoF fronthaul (IEEEtran, journal).

| File | What |
|---|---|
| `main.tex` | the paper; `\tbd{...}` (red) marks what waits for a result |
| `refs.bib` | the references |
| `Makefile` | `make -C paper`: `main.pdf`, built with latexmk in the TeX Live container `texlive/texlive:latest-full` (arm64 and amd64; nothing installed on the machine); `make -C paper clean` |

The link is the testbed's (the `arof-pipeline` repository) with the carrier at 28 GHz
(`link.f_rf` and `link.wireless.f_rf` in the run's settings); its numbers in the text come from
those settings and from the runs (`docs/testbed/jetson/HOOKUP.md`).
