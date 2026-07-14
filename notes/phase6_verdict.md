# Phase 6 verdict — MH45 at Re = 20,000

LBM-LES sweep: alpha 0-10 deg, chord 400 cells, Smagorinsky Cs = 0.14, momentum-exchange forces, error bars = std-dev over the measurement window.

| quantity | expected agreement? | achieved | verdict |
|---|---|---|---|
| lift-curve slope (pre-stall) | yes, within ~15% of XFOIL | ours: 6.76/rad (thin-airfoil ref: 6.28/rad) | PENDING — needs user XFOIL polars at data/xfoil_mh45_re20k.csv |
| drag level | NO — expected high (staircase + ~3-cell BL) | Cd(0) = 0.0310 | PENDING XFOIL overlay |
| stall angle | NO — different transition physics | — | explicitly untrusted, by design |

Mean Cl oscillation (shedding) amplitude: 0.169; per-alpha Strouhal in the CSV.

Figures: validation/mh45_polar.png (Cl, Cd vs alpha), validation/mh45_staircase.png (the honest mask).
