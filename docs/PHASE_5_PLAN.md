# Phase 5 — Empirical Applications: Design Spec

**Date**: 2026-05-27
**Status**: design approved (vertical slice), implementation plan to follow
**Predecessor**: `docs/PHASE_3_SCAFFOLD_COMPLETE.md`
**Reference design**: `PROPOSED_METHODOLOGY.md` §10–§11; phase plan `brainstorm-read-proposed-methodology-md-humble-abelson.md` Phase 5.

---

## 0. Decision record (this session)

- The full Stage 1 identifiability gate (heavy 30×3 SMC run) is **skipped for now**
  by user decision. Phase 5 proceeds without it.
- Colab is **out of scope**. Everything runs locally on this 20-core machine
  (MATLAB R2026a at `/home/d0/MATLAB/R2026a/bin/matlab`, not on `PATH`).
- OOS forecasting for the SV/RECH models uses the **fixed-parameter rolling
  scheme** (PROPOSED_METHODOLOGY §9.4 sanity option), **not** per-step
  data-annealing SMC. Rationale below (§3).
- First deliverable is a **vertical slice on Stage 4 (VN-Index)**: the whole
  pipeline end-to-end on one market with a small model set, producing real
  scores and a results table, before scaling to all markets/models.

---

## 1. Goal

Produce the project's headline empirical result — a rigorous SV-DL-hybrid
volatility forecast for the **Vietnamese market (VN-Index)** with four
exogenous covariates — by running the full data→fit→forecast→score→report
pipeline end-to-end. Success means a `results/stage4/` table comparing the
proposed **SV-LT-RECH (SRN)** against baselines on the five predictive scores,
with the data fetch validated against the NEU-thesis descriptive anchors.

This vertical slice de-risks the heavier multi-market Stages 2 and 3, which
reuse the same machinery.

## 2. Scope

### 2.1 In scope (the slice)

- **Data**: VN-Index daily close (vnstock) + 4 daily covariates per NEU thesis
  §3.5.3 — WTI crude oil, USD/VND exchange rate, Bitcoin, gold. Period targeting
  ~Jan 2016–Jan 2025, ≈2,262 obs, 1,500 train / 762 test.
- **Models** (kept small for the slice):
  - `garch.fitGarch` GARCH(1,1)-t — baseline (exists, cheap).
  - `garch.fitGarch` GJR-GARCH(1,1)-t — leverage baseline (exists, cheap).
  - `models.SVLT` — SV + leverage + Student-t baseline (exists; in-sample SMC + rolling PF OOS).
  - `models.SVLTRECH` (SRN, 4 covariates) — **the proposed flagship** (exists).
- **OOS**: fixed-parameter rolling 1-step-ahead predictive (§3).
- **Scores**: PPS, QS(1%,5%), MSE, MAE, R²LOG (`+eval/`), QLIKE as primary loss;
  Diebold–Mariano + Model Confidence Set across the four models.
- **Volatility proxy**: squared returns `r_t²` (no RV for an emerging market —
  PROPOSED_METHODOLOGY §10.1; flag in the table).
- **Report**: `results/stage4/scores.csv`, `forecasts.csv`, a comparison table,
  and one forecast-interval figure (F1-style).

### 2.2 Explicitly deferred (NOT this slice)

- SVLTRECH-LSTM / -GRU variants (add after the SRN slice works; trivial swap).
- `RSVLTRECH` realized-vol variant (Stage 2 only; needs an RV panel).
- GARCH-X and RealGARCH baselines (Stage 2/3; not yet built).
- Per-step data-annealing SMC (`+inference/+smc/dataAnneal.m`) — only needed if
  we later want the paper-exact OOS scheme.
- Stages 2 (10 markets + RV) and 3 (5 markets + covariates).

## 3. OOS forecasting scheme for SV/RECH models

The proposed models are fit in-sample with `inference.smc.likelihoodAnneal`,
yielding `N` posterior particles `{θ^(j)}`. For genuine 1-step-ahead OOS
forecasts we use a **fixed-parameter rolling predictive**:

1. Thin the posterior to `J` particles (default `J = 200`) by systematic
   resampling on the posterior weights (uniform after the final anneal step).
2. For each `θ^(j)`, run **one** bootstrap PF over the *full* series
   `y_{1:T+T_test}` (train+test), reading off, at each `t`, the PF's own
   one-step predictive log-density `log p(y_t | y_{1:t-1}, θ^(j))` and the
   one-step predicted variance `v̂_t^(j) = E[exp(h_t) | y_{1:t-1}, θ^(j)]`.
3. Bayesian model-average across particles in log space:
   `log p(y_t|y_{1:t-1}) ≈ logsumexp_j(logPred^(j)) − log J`, and
   `v̂_t = mean_j v̂_t^(j)`.
4. Keep the test-window slice `t = T+1 … T+T_test` for scoring.

**Why this and not data-annealing.** Data-annealing re-runs an SMC sweep at
every OOS step → ~`T_test` × (full SMC) cost, the same wall that stalled the
Stage 1 gate. The rolling scheme costs `J` particle filters over the full
series — `J × (T·M)` ≈ 200 × 2000 × 200, seconds, not weeks. It marginalises
parameter uncertainty (so PPS/QS remain genuine density scores) while holding
the posterior fixed at its in-sample value, which PROPOSED_METHODOLOGY §9.4 / §9.7
explicitly lists as the computational-sanity OOS option ("full re-run every 100
new obs" is a future refinement, deferred). The GARCH baselines already use the
analogous fixed-parameter rolling filter (`garch.rollingForecast`), so the
comparison is apples-to-apples.

## 4. Architecture / files

### 4.1 New code

| Path | Purpose |
|---|---|
| `+inference/+pf/bootstrap.m` (extend) | add `opts.returnIncrements` → also return `T×1` per-step predictive log-density and `T×1` one-step predicted variance. Additive, default-off; existing scalar/`hFiltered` behaviour unchanged. |
| `+inference/+forecast/rollingPredictive.m` (new `+forecast` subpkg) | the §3 routine: fit-posterior in, thinned-particle rolling predictive out (`logPredDensity`, `varForecast` over the test window). Model-agnostic over any `models.Model`. |
| `scripts/run_stage4_vietnam.m` | orchestrator: load config → fetch+preprocess → fit 4 models → OOS forecast → score (PPS/QS/MSE/MAE/R²LOG/QLIKE) → DM + MCS → write `results/stage4/`. |
| `config/experiments/stage4.yaml` | market = VN-Index, dates, covariate list, train/test split, per-model SMC sizes, `J` thinning, RNG seed. |
| `+data/loadStage4.m` (or reuse fetchers) | thin wrapper: `fetchVN` (VN-Index) + `fetchFRED`/`fetchYahoo` (oil/USD-VND/BTC/gold), align calendars, forward-fill non-trading days, drop missing-target rows, train-only standardise covariates. |

### 4.2 Tests

| Path | Asserts |
|---|---|
| `tests/tRollingPredictive.m` | on a synthetic SV series at fixed θ: (a) summed `logPredDensity` over the *whole* series matches `bootstrap`'s scalar `logLik` to PF MC noise; (b) test-window shapes correct; (c) `varForecast > 0`; (d) BMA over J particles is finite and stable. |
| `tests/tStage4Smoke.m` | tiny end-to-end on a cached/synthetic VN-like series (small N/M/J): pipeline runs, `scores.csv` well-formed, every score finite, GARCH and SVLTRECH both produce a row. Network-free (uses a fixture). |

### 4.3 Reused as-is

`+garch/*`, `+eval/*`, `+data/+preprocess/*`, `+data/+fetch/*`,
`+models/{SVLT,SVLTRECH}`, `inference.smc.likelihoodAnneal`,
`inference.pf.bootstrap` (post-extension), `+priors/*`.

## 5. Data flow

```
config/experiments/stage4.yaml
   │
   ▼  +data/loadStage4  (fetchVN + fetchFRED/fetchYahoo, cache-backed)
raw prices + covariates
   │
   ▼  +data/+preprocess  (log-returns ×100; stationarity check; μ=0/ARMA;
   │                       residual diagnostics; train-only standardise covs)
(y_train, y_test, Z_train, Z_test, splitIdx)
   │
   ├─► garch.fitGarch (GARCH-t, GJR-t) ─► garch.rollingForecast ─┐
   │                                                              │
   └─► smc.likelihoodAnneal (SVLT, SVLTRECH) ─► forecast.rollingPredictive ─┤
                                                                  ▼
                                                  per-model {logPredDensity, varForecast}
                                                                  │
                                                                  ▼  +eval (PPS/QS/MSE/MAE/R²LOG/QLIKE) + DM + MCS
                                                  results/stage4/{scores.csv, forecasts.csv, table, F1.png}
```

## 6. Validation anchors

- **Fetch sanity**: VN-Index `100·log` returns must reproduce NEU thesis
  Table 4.14 within tolerance — mean ≈ 0.0149, std ≈ 0.498, skew ≈ −0.987,
  **kurtosis ≈ 7.73**, n ≈ 2,262. A mismatch means the fetch (ticker, calendar,
  scaling) is wrong — halt and fix before fitting.
- **Diagnostics gate**: ARCH-LM must reject on `y` (justifies a volatility
  model); JB must reject (justifies Student-t). Recorded in the Stage 4 T1 row.
- **Direction check**: SVLTRECH log-marginal-likelihood and QLIKE should be at
  least competitive with GARCH-t. If SV-RECH is uniformly worse, report it
  honestly (per the plan's verification note) rather than re-tuning to win.

## 7. Error handling & risks

| Risk | Mitigation |
|---|---|
| **No network in this environment** → fetchers fail | First checkpoint is a fetch probe. If the live fetch fails, run the slice on a cached/synthetic VN-like fixture to validate the pipeline, and surface the network gap explicitly rather than fabricating data. The deliverable's real numbers wait for a successful fetch. |
| vnstock ticker/source drift | `fetchVN` is cache + manifest pinned; verify against the §6 anchor on first pull, then freeze the snapshot. |
| USD/VND daily series unavailable on FRED | Fall back to a documented alternative source or forward-fill from the lowest available frequency; document in `docs/data_sources.md`. |
| SV/RECH SMC still slow even in-sample (single fit, no OOS sweep) | One in-sample fit at N≈2000/M≈200/T≈2262 is ~minutes-to-hours unparallelised; acceptable for one market. If too slow, parallelise the `likelihoodAnneal` init loop with `parfor` (separately scoped; not required for the slice to be correct). |
| `parfor`/RNG reproducibility | Not used in the slice unless needed; if added, use per-worker substreams seeded from `baseSeed`. |
| Econometrics Toolbox absent | GARCH tests already `assumeTrue(ver('econ'))`; the runner skips GARCH rows with a logged warning if absent (it is present here). |

## 8. Acceptance criteria for the slice

1. `tRollingPredictive` + `tStage4Smoke` pass (added to the suite; full suite
   stays green).
2. `run_stage4_vietnam.m` completes locally and writes a well-formed
   `results/stage4/scores.csv` with one row per model × score.
3. The VN-Index fetch matches the §6 descriptive anchor (or, if offline, the
   pipeline completes on the fixture and the network gap is reported).
4. A comparison table (GARCH-t, GJR-t, SVLT, SVLTRECH-SRN) and one forecast
   figure are produced. Numbers are reported as-found, not tuned.

## 9. Out-of-scope follow-ons (after the slice)

Add LSTM/GRU variants → Stage 3 (5 markets, covariates) → Stage 2 (10 markets +
RV, which needs `RSVLTRECH` + GARCH-X + RealGARCH + the RV panel) → optional
per-step `dataAnneal` for paper-exact OOS → Stage 5 MIDAS-LASSO.
