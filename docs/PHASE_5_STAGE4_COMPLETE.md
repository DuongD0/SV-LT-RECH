# Phase 5 — Stage-4 Vietnam Vertical Slice — Completion Record

**Date**: 2026-05-27
**Plan**: `docs/PHASE_5_IMPLEMENTATION_PLAN.md`
**Design**: `docs/PHASE_5_PLAN.md`
**Predecessor**: `docs/PHASE_3_SCAFFOLD_COMPLETE.md`

This records the **code-complete + offline-verified** state of the Stage-4
vertical slice. Every piece of the data→fit→forecast→score→compare pipeline is
built and proven on a synthetic fixture; the **live VN-Index data run is
deferred** because this environment has no outbound network (see §4).

---

## 1. What was built

### 1.1 New / modified files (9)

```
+inference/+pf/bootstrap.m                (modified) added opts.returnIncrements ->
                                          per-step predictive log-density + one-step
                                          predicted variance (additive, default-off)
+inference/+forecast/rollingPredictive.m  (new) fixed-parameter rolling 1-step
                                          Bayesian predictive over a thinned posterior
+experiments/runStage4.m                  (new) Stage-4 pipeline: fit 4 models,
                                          OOS-forecast, score, DM + MCS
+data/loadStage4.m                        (new) network loader: VN-Index + 4 covariates
scripts/run_stage4_vietnam.m              (new) thin entry point
config/experiments/stage4.yaml            (new) Stage-4 config
tests/tBootstrapIncrements.m              (new) PF increment correctness (2 tests)
tests/tRollingPredictive.m                (new) rolling predictive (2 tests)
tests/tStage4Smoke.m                      (new) network-free end-to-end (1 test)
```

### 1.2 Method implemented

- **PF predictive increments** — `bootstrap` now optionally returns
  `logPredDensity(t) = log p(y_t | y_{1:t-1}, theta)` and
  `varForecast(t) = E[exp(h_t) | y_{1:t-1}, theta]` (the one-step predicted
  variance, since `Var(y_t | h_t) = exp(h_t)` for the scaled-t observation).
- **Fixed-parameter rolling predictive** (PROPOSED_METHODOLOGY §9.4) — fit the
  in-sample posterior once, thin to `J` particles, run one PF over the full
  series per particle, and Bayesian-model-average per-step:
  `log p(y_t|y_{1:t-1}) ≈ logsumexp_j(logPred^(j)) − log J`, `v̂_t = mean_j v̂_t^(j)`.
- **Stage-4 pipeline** — GARCH-t and GJR-t via `garch.fitGarch` +
  `garch.rollingForecast`; SVLT and SVLTRECH (SRN, covariates) via
  `likelihoodAnneal` + `rollingPredictive`. Scores: PPS, QS(1%,5%), MSE, MAE,
  R²LOG, QLIKE; squared returns as the volatility proxy; Diebold-Mariano vs the
  proposed model + Model Confidence Set on per-step QLIKE.

## 2. Test results

Full suite via `runtests('tests')`:

```
SUITE PASSED=112 FAILED=0 INCOMPLETE=0 of 112
```

New-file breakdown:

| Suite | # tests | Status |
|---|---|---|
| `tBootstrapIncrements` | 2 | PASS (increments sum to scalar logLik; 2-output contract intact) |
| `tRollingPredictive` | 2 | PASS (shapes/positivity; BMA within per-particle range) |
| `tStage4Smoke` | 1 | PASS (full 4-model pipeline on a synthetic 2-covariate fixture) |

PF regression guards (`tParticleVsGrid`, `tParticleAuxDefault`) stayed green
after the `bootstrap.m` change. All four new MATLAB source files are
`checkcode`-clean.

## 3. Acceptance criteria (PHASE_5_PLAN §8)

| Criterion | Status |
|---|---|
| 1. New tests + full suite pass | **PASS** (112/112) |
| 2. `run_stage4_vietnam.m` writes well-formed `scores.csv` | **Deferred** — needs network (code proven via smoke fixture) |
| 3. VN-Index fetch matches the descriptive anchor (kurtosis ≈ 7.73) | **Deferred** — needs network |
| 4. Comparison table + reported as-found | **Deferred** — needs network |

## 4. Why the live run is deferred — no network

A FRED reachability probe timed out:

```
fetchFRED('DCOILWTICO', ...) -> connection ... timed out after 60.000 seconds
```

This environment has no outbound network, so `data.loadStage4` (which pulls
VN-Index via `vnstock`, oil/USD-VND/gold via FRED, BTC via Yahoo) cannot fetch
real data here. Per `docs/PHASE_5_PLAN.md` §7, we do **not** fabricate data: the
pipeline's correctness is established on the synthetic fixture, and the live run
is the single remaining step for a connected environment.

`fetchYahoo`'s output columns (`date, open, high, low, close, adjClose,
volume`) were verified to match `loadStage4`'s usage; `fetchFRED`/`fetchVN`
signatures were read and matched. So the loader is expected to work once
connectivity is available.

## 5. To finish on a connected machine

```bash
cd /home/d0/projects/finance_eng
source setup.sh
/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); run('scripts/run_stage4_vietnam.m')" 2>&1 | tee results/stage4/run.log
```

Then verify the printed VN-Index descriptive line against the NEU-thesis anchor
(mean ≈ 0.0149, std ≈ 0.498, skew ≈ −0.987, kurtosis ≈ 7.73, n ≈ 2,262) before
trusting the scores. If `DEXVNUS` (USD/VND) is unavailable on FRED, substitute a
documented alternative and note it in `docs/data_sources.md`.

## 6. Known residual items

- **USD/VND source**: `loadStage4` uses FRED `DEXVNUS`; verify availability on
  first live pull.
- **Follow-ons** (unchanged from the design spec §9): add SVLTRECH-LSTM/-GRU to
  the Stage-4 comparison; Stage 3 (5 markets + covariates); Stage 2 (10 markets
  + RV, which needs `RSVLTRECH` + GARCH-X + RealGARCH + an RV panel); optional
  per-step `dataAnneal` for paper-exact OOS; Stage 5 MIDAS-LASSO.
