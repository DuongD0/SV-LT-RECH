# Phase 6 — Reporting & Figures Infrastructure — Completion Record

**Date**: 2026-05-29
**Plan**: `docs/superpowers/plans/2026-05-28-phase6-reporting-figures.md`
**Design**: `docs/superpowers/specs/2026-05-28-phase6-reporting-figures-design.md`
**Predecessor**: `docs/PHASE_5_STAGE4_COMPLETE.md`

This records the **code-complete + offline-verified** state of the Phase-6
paper-deliverable layer: tables T1–T7 and figures F1–F4
(`PROPOSED_METHODOLOGY.md` §12), driven by a self-describing results bundle
persisted by an expanded 6-model Stage-4 pipeline, emitted as CSV + LaTeX +
PNG/PDF, and proven end-to-end on a synthetic results bundle. The **live
VN-Index run remains deferred** (no outbound network here — unchanged from
Phase 5).

---

## 1. What was built

### 1.1 New / modified files

```
+data/+preprocess/bdsTest.m            (new) compact correlation-integral BDS test (T1 nonlinearity)
+experiments/stage4Models.m           (new) 6-model registry (single source of truth)
+models/SVLTRECH.m                     (mod) added omegaPath() interpretability method (F4)
+experiments/runStage4.m               (rewrite) registry loop -> 6 models -> assemble + persist bundle.mat
+report/assertBundle.m                 (new) bundle-shape validator
+report/latexTable.m                   (new) table -> booktabs LaTeX (bold-best, NaN->'--')
+report/withDefaults.m                 (new) shared figure-option defaults
+report/saveFigure.m                   (new) write <stem>.png + <stem>.pdf
+report/tableT1descriptives.m          (new) T1 descriptives + ADF/PP/KPSS + JB/ARCH-LM/BDS
+report/tableT2posteriors.m            (new) T2 posterior mean(std) per parameter per model
+report/tableT3logml.m                 (new) T3 log marginal likelihoods
+report/tableT4scores.m                (new) T4 PPS/QS1/QS5/MSE/MAE/R2LOG/QLIKE
+report/tableT5dm.m                    (new) T5 Diebold-Mariano vs proposed
+report/tableT6mcs.m                   (new) T6 Model Confidence Set membership
+report/tableT7residuals.m             (new) T7 std-residual moments + LB-Q^2
+report/figForecastBands.m             (new) F1 95% one-step bands over OOS returns
+report/figQQ.m                        (new) F2 standardized-residual QQ vs N(0,1)
+report/figCovariatePosterior.m        (new) F3 v_z posterior densities
+report/figOmegaState.m                (new) F4 omega_t recurrent-state path
scripts/make_paper_tables.m            (new) bundle -> all 7 tables (CSV+LaTeX) + 4 figures (PNG+PDF)
scripts/run_stage4_demo.m              (new) offline synthetic end-to-end demo
tests/fixtures/syntheticBundle.m       (new) in-memory bundle fixture for report tests
tests/tBdsTest.m                       (new) BDS iid-vs-dependent sanity (3)
tests/tStage4Models.m                  (new) registry shape (3)
tests/tOmegaPath.m                     (new) omegaPath shape + omega_1==beta_0 (2)
tests/tReportTables.m                  (new) T1-T7 builders + latexTable + orchestrator (10)
tests/tReportFigures.m                 (new) F1-F4 emit non-empty PNG+PDF (4)
tests/tStage4Smoke.m                   (mod) expanded to 6 models + bundle assertions (1)
```

### 1.2 Method implemented

- **Results bundle** (design spec §4.1) — `runStage4` now assembles a single
  self-describing struct (`.meta/.data/.descriptives/.models/.comparison`) saved
  to `results/<exp>/bundle.mat`. Per-model records carry posterior particles,
  log-marginal-likelihood, in-sample standardized residuals, the ω-state path
  (RECH models), and OOS forecasts. The `+report` layer is a **pure consumer** of
  the saved bundle — no inference is re-run.
- **6-model comparison** — GARCH-t, GJR-t (Econometrics MLE) + SVLT,
  SVLTRECH-SRN (proposed), SVLTRECH-LSTM (co-primary), SVLTRECH-GRU (sensitivity),
  driven by `experiments.stage4Models()`.
- **Reporting** — `make_paper_tables(resultDir)` builds T1–T7 (CSV + booktabs
  LaTeX with bold-best-value, NaN→`--`) and F1–F4 (headless PNG + vector PDF).
- **BDS** — compact Brock-Dechert-Scheinkman correlation-integral statistic added
  so T1 matches §12.

## 2. Test results

Full suite via `runtests('tests')`:

```
FULLSUITE PASS=134 FAIL=0 INCOMPLETE=0 of 134
```

New / changed Phase-6 tests (23): `tBdsTest` (3), `tStage4Models` (3),
`tOmegaPath` (2), `tReportTables` (10), `tReportFigures` (4), `tStage4Smoke` (1,
expanded to 6 models + bundle contract). All MATLAB source `checkcode`-clean.

## 3. Offline demonstration (synthetic, as-found)

`scripts/run_stage4_demo.m` runs the full chain on a synthetic VN-like series
drawn from the SVLTRECH-SRN data-generating process (K=2 covariates, T=320,
split 240, N=300 / M=40 / J=30), then `make_paper_tables` over the persisted
bundle. Outputs in `results/stage4_demo/`:
`bundle.mat`, `scores.csv`, `tables/T1..T7.{csv,tex}`, `figures/F1..F4.{png,pdf}`.

**T4 predictive scores (reported as-found, not tuned):**

| Model | PPS | QS1 | QS5 | MSE | MAE | R²LOG | QLIKE | inMCS | logML |
|---|---|---|---|---|---|---|---|:--:|---|
| GARCH-t | 1.705 | 0.0401 | 0.1449 | **0.848** | **0.773** | **5.57** | 1.603 | yes | — |
| GJR-t | 1.711 | 0.0392 | 0.1435 | 1.029 | 0.862 | 6.08 | 1.614 | yes | — |
| SVLT | 1.707 | 0.0573 | 0.1875 | 2.517 | 1.437 | 9.56 | 2.035 | no | −414.41 |
| **SVLTRECH-SRN** | **1.696** | **0.0333** | **0.1366** | 0.864 | 0.788 | 5.66 | **1.558** | yes | **−411.92** |
| SVLTRECH-LSTM | 1.701 | 0.0383 | 0.1376 | 1.033 | 0.881 | 6.33 | 1.606 | yes | −413.03 |
| SVLTRECH-GRU | 1.701 | 0.0383 | 0.1437 | 0.891 | 0.806 | 5.82 | 1.582 | no | −413.64 |

Reading (synthetic data only — **not** an empirical claim): on its own DGP the
proposed **SVLTRECH-SRN** attains the best density/VaR scores (PPS, QS1, QS5,
QLIKE) and the highest SV log marginal likelihood; GARCH-t wins the
point-volatility metrics (MSE/MAE/R²LOG) against the squared-return proxy. The
Model Confidence Set retains {GARCH-t, GJR-t, SVLTRECH-SRN, SVLTRECH-LSTM}.
This validates the pipeline mechanics; it is not evidence about real markets.

## 4. Acceptance criteria (plan §8)

| Criterion | Status |
|---|---|
| 1. `runStage4` (6-model) persists a bundle; smoke + full suite green | **PASS** (134/134) |
| 2. Each `report.tableT1..T7` returns a well-formed table; `latexTable` bolds best | **PASS** |
| 3. Each `report.fig*` writes non-empty PNG + PDF | **PASS** |
| 4. `make_paper_tables` emits all 7 tables (CSV+LaTeX) + 4 figures | **PASS** (`results/stage4_demo/`) |
| 5. `bdsTest` passes iid-vs-dependent sanity | **PASS** |
| 6. Completion record written | **PASS** (this file) |

## 5. To obtain real numbers on a connected machine

```bash
cd /home/d0/projects/finance_eng
source setup.sh
matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); run('scripts/run_stage4_vietnam.m')"
matlab -batch "addpath(pwd); addpath('scripts'); make_paper_tables('results/stage4')"
```

`run_stage4_vietnam.m` already calls `experiments.runStage4` with
`writeOutputs=true`, so after the Task-4 rewrite it persists `bundle.mat`; the
only added step for the paper deliverable is `make_paper_tables('results/stage4')`.
Verify the printed VN-Index descriptive line against the NEU-thesis anchor
(kurtosis ≈ 7.73, std ≈ 0.498, n ≈ 2,262) before trusting the numbers.

## 6. Notes / residual items

- **`run()` changes CWD.** MATLAB's `run('scripts/X.m')` cd's into `scripts/`
  during execution, so a bare relative output path lands under `scripts/`.
  `run_stage4_demo.m` anchors `outDir` to the project root via
  `fileparts(fileparts(mfilename('fullpath')))`. `scripts/run_stage4_vietnam.m`
  should adopt the same anchoring before the live run (or be invoked so its CWD
  is the project root) to avoid misplacing `results/stage4`.
- **omegaPath (F4)** is implemented for the SRN flagship only; LSTM/GRU records
  carry an empty `omegaPath` and F4 omits them — intentional per the design spec.
- **GARCH posterior std** is NaN (MLE point estimate; the toolbox object exposes
  no SE here); `latexTable` renders NaN as `--`.
- **Deferred (unchanged):** live VN-Index fetch (no network), SMC
  parallelization, Stage 3 (5 markets), Stage 2 (10 markets + RV; needs
  RSVLTRECH + GARCH-X + RealGARCH + an RV panel), Stage 5 MIDAS-LASSO, and the
  prose paper draft.
