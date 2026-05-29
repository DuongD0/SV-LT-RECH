# Phase 6 — Reporting & Figures Infrastructure: Design Spec

**Date**: 2026-05-28
**Status**: design approved
**Predecessor**: `docs/PHASE_5_STAGE4_COMPLETE.md`
**Reference design**: `PROPOSED_METHODOLOGY.md` §10, §12, §13; phase plan
`brainstorm-read-proposed-methodology-md-humble-abelson.md` Phase 6.

---

## 0. Decision record (this session)

- **Phase 6 focus**: reporting + figures infrastructure — the paper-deliverable
  layer (tables T1–T7, figures F1–F4 of `PROPOSED_METHODOLOGY.md` §12). Not SMC
  parallelization, not Stage 5 MIDAS-LASSO, not the prose paper draft.
- **Completeness**: the full T1–T7 + F1–F4 set.
- **Output formats**: machine-readable CSV **and** publication LaTeX (booktabs,
  bold-best-value, MCSE in parentheses); figures as PNG **and** vector PDF.
- **Model set**: expand the Stage-4 comparison from 4 to **6** models — add
  `SVLTRECH-LSTM` (co-primary, §8.4) and `SVLTRECH-GRU` (sensitivity, §8.5) to
  the existing GARCH-t, GJR-t, SVLT, SVLTRECH-SRN. This folds in the Phase-6
  "robustness: GRU/LSTM variant" item.
- **Approach**: persist a complete, self-describing results bundle from the
  Stage-4 pipeline; the `+report` layer is a pure consumer of the saved bundle
  and never re-runs inference (Approach A; B/C rejected — see §2).
- **Environment reality (unchanged from Phase 5)**: no outbound network, so the
  live VN-Index run stays deferred. The whole reporting chain is built and
  proven **offline** on a synthetic results bundle. Real numbers wait for a
  connected machine.

---

## 1. Goal

Produce the project's paper-deliverable layer: a reproducible pipeline that
turns a saved Stage-4 results bundle into the complete set of reporting tables
(T1–T7) and figures (F1–F4) defined in `PROPOSED_METHODOLOGY.md` §12, emitted as
both CSV and publication-ready LaTeX/PDF. Success means `make_paper_tables.m`,
run over a results bundle, writes every table and figure with finite,
well-formed content, verified end-to-end on a synthetic bundle so a connected
machine need only run the live Stage-4 fetch + fit to obtain real numbers.

This completes the empirical-reporting machinery that Stages 2–4 all reuse.

## 2. Approach

**A — Persist-everything results bundle (chosen).** The Stage-4 pipeline computes
and persists a complete bundle (posteriors, log-marginal-likelihoods, per-model
standardized residuals, ω-state paths, OOS forecasts, per-step QLIKE, scores,
DM, MCS, in-sample descriptives/diagnostics). `+report` consumes the saved
bundle only. Single source of truth; reporting decoupled from inference and
cheap to re-run; matches §13 reproducibility outputs.

**B — Recompute in the report layer (rejected).** Persist only posteriors+fits;
report functions recompute residuals/ω/diagnostics on demand. Couples reporting
to the inference stack (models + PF + data), making table generation slow and
fragile.

**C — Thin subset from existing CSVs (rejected).** T3–T6 + F1 only from the
current `scores.csv`/`forecasts.csv`; the declined "scores subset" option.

## 3. Scope

### 3.1 In scope

- **Results-bundle schema + persistence** written by the Stage-4 pipeline.
- **Stage-4 model-set expansion 4 → 6** via a config-driven model registry that
  replaces the four copy-pasted per-model blocks in `+experiments/runStage4.m`.
- **`+report/` table builders** (T1–T7), one focused function per table.
- **`scripts/make_paper_tables.m`** orchestrator → CSV + LaTeX.
- **`+report/latexTable.m`** booktabs renderer (bold-best, MCSE-in-parens).
- **`+report/` figure builders** (F1–F4) → PNG + vector PDF.
- **Compact BDS test** (`+data/+preprocess/bdsTest.m`) so T1 matches §12.
- **Tests** for every new unit + a full-suite regression guard.
- **Offline demonstration** producing a real synthetic `results/stage4_demo/`.

### 3.2 Explicitly out of scope

- Live VN-Index network run (no connectivity here — unchanged from Phase 5).
- SMC parallelization (`+utils/parPool.m`, `parfor`).
- Stage 5 MIDAS-LASSO (`+inference/+mcmc/proximalStep.m`, `run_stage5_midas.m`).
- Stages 2 (10 markets + RV) and 3 (5 markets) experiment runs.
- The prose paper draft.

## 4. Architecture / files

### 4.1 Results-bundle schema

A single `bundle` struct, saved as `results/<exp>/bundle.mat`, with:

| Field | Content |
|---|---|
| `.meta` | experiment name, date, `cfg` snapshot, `baseSeed`, model registry list |
| `.data` | `y`, `Z`, `splitIdx`, `dates` (datetime or numeric index), `covNames` |
| `.descriptives` | T1 record: returns moments + `stationarityTests` + `residualDiagnostics` (+ BDS) on `y` |
| `.models(m)` | per-model struct array, fields below |
| `.comparison` | `scores` (struct array), `dmVsProposed`, `mcs`, per-step `qlAll`, `modelNames` |

Per-model `.models(m)` fields:

| Field | GARCH/GJR | SVLT / SVLTRECH-{SRN,LSTM,GRU} |
|---|---|---|
| `.name`, `.type` | yes | yes |
| `.paramNames`, `.posteriorMean`, `.posteriorStd` | from fit (point + se) | from `result.theta` columns |
| `.theta` | empty | N×K posterior particles (for F3) |
| `.logMarginalLik` | NaN | `result.logMarginalLik` |
| `.nu` | `fit.nu` | `mean(theta(:,5))` |
| `.sigma2InSample` | σ²-path | `exp(hFiltered)` from PF `returnPath` |
| `.stdResid` | `y./sqrt(σ²)` → t-CDF→Φ⁻¹ | `Φ⁻¹(F_ν(y_t/σ̂_t))` |
| `.omegaPath` | empty | ω_t path (RECH only; SVLT → empty) |
| `.varForecast`, `.logPredDensity` | test-window | test-window |

The bundle is self-describing: every table/figure builder reads only from it.

### 4.2 New / changed code

| Path | Responsibility |
|---|---|
| `+experiments/runStage4.m` (refactor) | model-registry loop (6 models); compute + assemble the bundle; keep the public `out` contract a superset of today's fields (back-compatible with `tStage4Smoke`) |
| `+experiments/stage4Models.m` (new) | returns the model registry: per-entry name, kind (`garch`/`sv`), constructor handle, leverage — single place to add/remove models |
| `+report/tableT1descriptives.m` | descriptives + ADF/PP/KPSS + JB/ARCH-LM/BDS per series |
| `+report/tableT2posteriors.m` | posterior mean (std) of all parameters per model |
| `+report/tableT3logml.m` | log marginal likelihoods across SV/RECH models |
| `+report/tableT4scores.m` | PPS/QS(1%,5%)/MSE/MAE/R²LOG per model |
| `+report/tableT5dm.m` | Diebold–Mariano statistic + p-value, each baseline vs proposed |
| `+report/tableT6mcs.m` | MCS membership + p-values + elimination order |
| `+report/tableT7residuals.m` | std-resid mean/std/skew/kurt + LB-Q² per model |
| `+report/latexTable.m` | `table` → booktabs LaTeX (bold-best column rule, MCSE parens) |
| `+report/assertBundle.m` | validate required bundle fields up front with a clear error; called once by `make_paper_tables` |
| `+report/figForecastBands.m` | F1: 95% one-step bands over OOS returns |
| `+report/figQQ.m` | F2: standardized-residual QQ vs N(0,1), per model |
| `+report/figCovariatePosterior.m` | F3: v_z posterior densities (kernel density) |
| `+report/figOmegaState.m` | F4: ω_t recurrent-state interpretability path |
| `+data/+preprocess/bdsTest.m` | compact correlation-integral BDS statistic |
| `scripts/make_paper_tables.m` | load bundle → all 7 tables (CSV+LaTeX) → all 4 figures (PNG+PDF) → `results/<exp>/tables/`, `results/<exp>/figures/` |

### 4.3 Reused as-is

`+eval/*` (T4/T5/T6 metrics), `+data/+preprocess/{stationarityTests,
residualDiagnostics}` (T1/T7), `inference.smc.likelihoodAnneal`,
`inference.pf.bootstrap` (`returnPath` for in-sample σ²/residuals),
`inference.forecast.rollingPredictive`, `+garch/*`, `+diagnostics/posteriorTrace`
(figure-handle pattern), `+models/{SVLT,SVLTRECH,SVLTLSTMRECH,SVLTGRURECH}`.

## 5. Data flow

```
results/<exp>/bundle.mat   (written by experiments.runStage4)
   │
   ▼  scripts/make_paper_tables.m  (load bundle)
   │
   ├─► report.tableT1..T7  ─► table objects ─► CSV  +  report.latexTable ─► .tex
   │
   └─► report.figForecastBands / figQQ / figCovariatePosterior / figOmegaState
                                              ─► results/<exp>/figures/F{1..4}.{png,pdf}
```

## 6. Validation anchors

- **Table-shape contracts**: T4 has one row per model × {PPS,QS1,QS5,MSE,MAE,
  R²LOG}; T5 has `nModels-1` pairwise rows; T6 lists every model with a 0/1
  membership flag; T2 has one row per parameter per model. All values finite on
  the synthetic bundle.
- **Bold-best logic**: `latexTable` marks the min (PPS/QS/MSE/MAE/R²LOG/QLIKE) or
  max (log-ML) per metric column; unit-tested against a hand-built table.
- **BDS sanity**: does not over-reject at nominal size on iid Gaussian input;
  rejects strongly on a GARCH-simulated (nonlinear-dependent) series.
- **Figure emission**: each `fig*` returns a valid handle and writes non-empty
  PNG + PDF; no exceptions on the synthetic bundle.
- **Direction check (reported as-found)**: whether SVLTRECH variants beat the
  GARCH baselines on the synthetic demo is reported honestly, not tuned
  (`PROPOSED_METHODOLOGY` verification note).

## 7. Error handling & risks

| Risk | Mitigation |
|---|---|
| `runStage4` refactor breaks the existing 4-model contract / `tStage4Smoke` | Keep `out` a superset (same field names/order for the first 4 models); run `tStage4Smoke` as a regression guard; expand it to 6 models only after the 4-model form stays green |
| 6-model synthetic run too slow even at small N/M | Demo uses tiny `N/M/J` (e.g. 250/40/20) and a short series; SV/RECH fits are seconds at that size |
| `bundle.mat` schema drift vs builders | `report.assertBundle.m` checks required top-level fields up front with a clear error; builders then read named fields directly |
| LaTeX/`booktabs` rendering edge cases (NaN, empty cells) | `latexTable` renders NaN/empty as `--`; unit-tested |
| BDS implementation correctness | TDD against the iid-vs-GARCH sanity case; reference Brock-Dechert-Scheinkman correlation-integral definition; keep `m`, `ε=0.5σ` per §4 |
| Econometrics Toolbox absence | T1/T7 builders that call `lbqtest`/`archtest` guard with `assumeTrue(ver('econ'))` in tests; `make_paper_tables` logs a warning and emits the toolbox-free tables if absent (present here) |
| Figure functions need a display | Use `figure('Visible','off')` + `exportgraphics`; never require a screen |

## 8. Acceptance criteria

1. `runStage4` (6-model) persists a complete `bundle.mat`; `tStage4Smoke`
   (expanded) and the full suite stay green.
2. Each `report.tableT1..T7` returns a well-formed `table` from the synthetic
   bundle; `latexTable` renders booktabs with correct bold-best marking.
3. Each `report.fig*` writes a non-empty PNG + PDF without error.
4. `scripts/make_paper_tables.m` over the synthetic bundle emits all seven
   tables (CSV + LaTeX) and all four figures into `results/stage4_demo/`.
5. `bdsTest` passes its iid-vs-dependent sanity test.
6. A completion record (`docs/PHASE_6_COMPLETE.md`) summarises files added, test
   tally, the synthetic demo's table/figure inventory, and the deferred live run.

## 9. Out-of-scope follow-ons (after this slice)

Live VN-Index run on a connected machine → SMC parallelization → Stage 3
(5 markets) → Stage 2 (10 markets + RV; needs `RSVLTRECH` + GARCH-X + RealGARCH +
RV panel) → Stage 5 MIDAS-LASSO → prose paper draft.
