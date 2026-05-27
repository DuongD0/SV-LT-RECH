# Phase 1 & Phase 2 — Completion Record + Phase 3 Guidelines

**Project**: SV-LT-RECH — Stochastic Volatility with Leverage, Student-t innovations, and an RNN-augmented log-variance state.
**Reference design**: `PROPOSED_METHODOLOGY.md` (in repo root).
**Reference papers**: `methodlogy_ref/FRL_GARCH_DL (1).pdf_*.md` and `methodlogy_ref/2. Building statistical machine learning models for volatility forecasting (2).pdf_*.md`.

This document is the authoritative inventory of what is built and the operational specification for Phase 3.

---

## 0. Environment (verified working)

| Component | Version | Location |
|---|---|---|
| Python venv | 3.13.12 | `.venv/` |
| `numpy / pandas / pyyaml / requests` | 2.2 / 2.3 / 6.0 / 2.34 | venv |
| `yfinance / pandas-datareader / vnstock` | 1.3 / 0.10 / 4.0 | venv |
| R (conda) | 4.5.3 | `~/miniconda3/envs/sv-lt-rech-r/` |
| `stochvol` | 3.2.9 | conda env |
| MATLAB | R2026a Update 1 | `~/MATLAB/R2026a/` |
| MATLAB toolboxes | Econometrics + Stats/ML + Parallel + Optimization | — |

`source setup.sh` activates all three (venv + conda + MATLAB on `PATH`).

---

## 1. Phase 1 — SMC engine + plain SV sanity (COMPLETE)

### 1.1 Files

| Path | Purpose | LoC |
|---|---|---|
| `+utils/logsumexp.m` | numerically stable `log(sum(exp(x)))` | ~25 |
| `+utils/log1pexp.m` | softplus with four-regime stability | ~25 |
| `+utils/reproducibility.m` | threefry RNG seed (parfor-safe) | ~15 |
| `+utils/loadConfig.m` | YAML reader (PyYAML via interop) | ~75 |
| `+models/Model.m` | abstract base class — every model inherits this | ~80 |
| `+priors/logPriorSV.m` | `mu`/`phi`/`sigma_eta` prior log-density | ~35 |
| `+models/SV.m` | plain SV (Kim–Shephard–Chib 1998) | ~85 |
| `+inference/+pf/bootstrap.m` | bootstrap PF in log-weight space | ~120 |
| `+inference/+pf/resampleSystematic.m` | systematic resampling | ~25 |
| `+inference/+pf/resampleStratified.m` | stratified resampling | ~25 |
| `+inference/+smc/ess.m` | effective sample size from log weights | ~20 |
| `+inference/+smc/adaptiveTemperature.m` | bisection for next `a_k` | ~55 |
| `+inference/+smc/likelihoodAnneal.m` | full likelihood-annealing SMC | ~150 |
| `+inference/+mcmc/rwMetropolis.m` | RWM rejuvenation kernel | ~80 |
| `scripts/calibrate_against_stochvol.R` | R bridge: `svsample()` | ~50 |
| `scripts/calibrate_against_stochvol.m` | MATLAB driver for cross-engine check | ~75 |
| `scripts/run_phase1_smc_sanity.m` | runs all tests + R cross-check | ~30 |
| `config/smc_defaults.yaml` | SMC hyperparameters | ~25 |
| `config/priors.yaml` | per-model prior specs (human-readable) | ~35 |

### 1.2 Mathematical specification

**Plain SV (Kim–Shephard–Chib 1998):**
```
r_t = exp(h_t/2) * eps_t,        eps_t ~ N(0, 1)
h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t,    eta_t ~ N(0, 1),    |phi| < 1
```

**Priors (§9.1):**
- `mu  ~ N(0, 10^2)`
- `(1+phi)/2 ~ Beta(20, 1.5)`
- `sigma_eta ~ Half-Cauchy(0, 1)`

**Likelihood-annealing SMC (§9.2):**
Sample sequentially from `pi_k(theta) ∝ p(theta)·p(y|theta)^{a_k}` with `0 = a_1 < ... < a_K = 1`, chosen so the conditional ESS at each step equals `N/2`. Each step:
1. Reweight by `L(theta)^{a_k − a_{k−1}}`.
2. Resample (systematic) when `ESS < 0.5·N`.
3. Apply `nSweeps` RWM moves at temperature `a_k`.

**Log-marginal-likelihood byproduct:**
```
log p(y) = sum_k log sum_i w_{k-1,i} * L_i^{a_k - a_{k-1}}
```

### 1.3 Tests (Phase 1)

| Test | Assertion | Verified |
|---|---|---|
| `tPriorsDensity/svPriorMatchesHand` | closed-form prior pdf at θ=[0, 0.9, 0.5] matches `logPriorSV` to 1e-10 | ✓ |
| `tPriorsDensity/outOfSupportReturnsNegInf` | -Inf for `phi∉(-1,1)` or `sigma_eta≤0` | ✓ |
| `tPriorsDensity/priorSamplesAreInSupport` | 500 prior draws all in support | ✓ |
| `tParticleVsGrid/plainSvAgreesWithGrid` | PF log-lik vs 1000-point grid filter to ±0.5 nats | ✓ |
| `tResamplingInvariance/{systematic,stratified}PreservesMean` | resampled mean → weighted mean over 2000 draws | ✓ |
| `tResamplingInvariance/indicesInRange` | resampling returns valid 1..N indices | ✓ |
| `tEssGradient/essMonotoneInTemperature` | ESS monotone non-increasing in `a` | ✓ |
| `tEssGradient/essAtZeroIsN` | uniform weights → ESS = N | ✓ |
| `tEssGradient/essBoundedBetweenOneAndN` | random weights → ESS ∈ [1, N] | ✓ |
| `tPriorPredictive/lowVolDataDrivesMuNegative` | sd-0.05 series → posterior `mu < -2` | ✓ |
| `tPriorPredictive/highVolDataDrivesMuPositive` | sd-3 series → posterior `mu > 1` | ✓ |
| `tPriorPredictive/allParticlesInSupport` | every posterior particle is in prior support | ✓ |

**Cross-engine calibration** (`scripts/calibrate_against_stochvol.m`):
On a 1000-day simulated SV path with `θ_true = [0.0, 0.95, 0.4]`, our SMC posterior means agree with R `stochvol::svsample` within 1 σ on every parameter:

| Param | truth | stochvol mean (sd) | ours mean (sd) |
|---|---|---|---|
| `mu` | 0.0 | 0.168 (0.291) | 0.193 (0.369) |
| `phi` | 0.95 | 0.939 (0.015) | 0.941 (0.017) |
| `sigma_eta` | 0.4 | 0.517 (0.047) | 0.524 (0.049) |

### 1.4 Bugs caught and fixed during Phase 1

- `tPriorPredictive` was a moment-recovery test, which is ill-defined under a Half-Cauchy prior (no mean). Rewrote as a likelihood-direction test that catches sign-flip bugs cleanly.
- `adftest` in R2026a no longer returns `reg.lags`. Removed that field from `+data/+preprocess/stationarityTests.m`.

---

## 2. Phase 2 — data + diagnostics (COMPLETE)

### 2.1 Files

**Data fetch + preprocess**
| Path | Purpose |
|---|---|
| `+data/+fetch/manifest.yaml` | pinned URLs + sha256 + snapshot dates for every external source |
| `+data/+fetch/cacheFetch.m` | manifest-aware filesystem cache wrapper |
| `+data/+fetch/fetchFRED.m` | FRED `fredgraph.csv` endpoint |
| `+data/+fetch/fetchYahoo.m` | yfinance via Python interop |
| `+data/+fetch/fetchRealizedLibrary.m` | Liu et al. (2025) RV panel + OMI fallback |
| `+data/+fetch/fetchVN.m` | vnstock via Python interop (VN-Index, HNX-Index) |
| `+data/+preprocess/standardize.m` | leakage-safe in-sample z-score |
| `+data/+preprocess/trainTestSplit.m` | chronological split, never shuffled |
| `+data/+preprocess/stationarityTests.m` | ADF + PP + KPSS combined verdict |
| `+data/+preprocess/residualDiagnostics.m` | LB + McLeod-Li + ARCH-LM + JB + moments |
| `+data/+preprocess/meanEquation.m` | μ=0 default; ARMA(p,q) by BIC fallback |

**Evaluation**
| Path | Purpose |
|---|---|
| `+eval/pps.m` | partial predictive score (mean negative log predictive density) |
| `+eval/quantileScore.m` | quantile / VaR loss (Taylor 2019) |
| `+eval/mseMae.m` | MSE + MAE of `sqrt(RV) − v_hat` |
| `+eval/r2log.m` | Hansen–Lunde `R²_LOG` |
| `+eval/qlike.m` | quasi-likelihood loss (Patton 2011) — primary loss for DM and MCS |
| `+eval/dieboldMariano.m` | DM test with Newey–West HAC variance |
| `+eval/modelConfidenceSet.m` | Hansen–Lunde–Nason 2011 MCS via Politis–Romano stationary bootstrap |

### 2.2 Key design contracts

- **No data leakage.** `standardize` uses in-sample statistics only; covariates standardised before they touch any model.
- **Train/test split is chronological.** No shuffling — destroys volatility clustering.
- **Mean-equation gate is data-driven.** Ljung-Box on raw returns at lag 10 decides between `μ=0` and ARMA fallback; BIC chooses `(p, q) ∈ {0..3}²`.
- **Manifest is authoritative.** Every external pull is checksummed against `manifest.yaml`. `pending` becomes a real SHA-256 on first successful fetch; subsequent fetches abort on mismatch.
- **Network-dependent fetchers are NOT unit-tested.** Their tests verify schema-shape on synthetic payloads. The cache-and-checksum mechanism IS unit-tested with a synthetic in-memory fetcher.

### 2.3 Tests (Phase 2)

| Test | Assertion |
|---|---|
| `tDataPreprocessAndEval/standardizeUsesTrainStats` | test slice uses train mean/std, not test stats |
| `tDataPreprocessAndEval/standardizeHandlesConstantColumn` | constant column → all-zero output + `constantCol` flag |
| `tDataPreprocessAndEval/splitChronological` | 75/25 split preserves order |
| `tDataPreprocessAndEval/stationarityFlagsReturnsButNotPrices` | returns I(0), prices I(1) |
| `tDataPreprocessAndEval/diagnosticsCatchArchEffect` | ARCH-LM rejects on SV-style series, fails to reject on iid normal |
| `tDataPreprocessAndEval/dmRejectsWhenLossDiffers` | DM rejects at α=0.05 when one model has uniformly larger loss |
| `tDataPreprocessAndEval/dmAcceptsWhenLossEqual` | DM statistic = 0, p = 1 for identical loss series |
| `tCacheAndFetch/freshThenCachedHit` | first call writes cache, second call reads it |
| `tCacheAndFetch/forceBypassesCache` | `opts.force` triggers fresh fetch |
| `tCacheAndFetch/checksumMismatchAborts` | wrong manifest sha256 → error |
| `tCacheAndFetch/unknownEntryFails` | unknown name in section → error |
| `tMeanEquationAndMCS/zeroMeanOnWhiteNoise` | i.i.d. normal → `usedARMA == false` |
| `tMeanEquationAndMCS/armaFitsAr1Process` | AR(1) series → `usedARMA == true`, LB p-value rises on residuals |
| `tMeanEquationAndMCS/forceZeroOverridesGate` | `force='zero'` returns raw input even on autocorrelated data |
| `tMeanEquationAndMCS/mcsKeepsBestEliminatesWorst` | 3-model setup with one terrible model → that one is eliminated |
| `tMeanEquationAndMCS/mcsKeepsAllWhenEquallyGood` | exchangeable-loss models → all in set |
| `tMeanEquationAndMCS/mcsHandlesTwoModels` | minimum K=2 case works |

### 2.4 Data sources (live, manifest-pinned)

| Source | Coverage | Entry point | License/terms |
|---|---|---|---|
| Yahoo Finance | SPX/N225/DJI/DAX/FTSE/AORD/CAC40/BVSP/AEX/BFX, BTC-USD | `fetchYahoo` | personal/research use only |
| FRED | VIX, oil, gold, USD/VND, INDPRO, UNRATE, CPI | `fetchFRED` | public |
| Liu et al. 2025 RealRECH panel | 31-index RV through 2023 | `fetchRealizedLibrary` | check repo |
| OMI Wayback snapshot | RV pre-July 2022 | `fetchRealizedLibrary` fallback | archival |
| vnstock (VCI/TCBS) | VN-Index, HNX-Index | `fetchVN` | terms vary |

---

## 3. Test inventory snapshot

After Phase 1 + 2:

| Suite | # tests |
|---|---|
| `tEssGradient` | 3 |
| `tParticleVsGrid` | 1 |
| `tPriorPredictive` | 3 |
| `tPriorsDensity` | 3 |
| `tResamplingInvariance` | 3 |
| `tDataPreprocessAndEval` | 7 |
| `tCacheAndFetch` | 4 |
| `tMeanEquationAndMCS` | 6 |
| **Total** | **30** |

Run via `runtests('tests')` in MATLAB after `addpath(pwd); pyenv('Version', '.venv/bin/python')`. See `scripts/run_phase1_smc_sanity.m` for the full pipeline (tests + R cross-check).

---

## 4. Phase 3 — simulation study + identifiability (NOT STARTED)

Phase 3 is the **go/no-go gate for everything downstream**. If `SV-LT-RECH` cannot recover known parameters on synthetic data, no real-data fit matters. Phase 3 produces this proof.

### 4.1 Files to create

```
+cells/
    srn.m                          ReLU simple-RNN cell forward pass
+models/
    SVt.m                          SV + Student-t innovations (§7.5b)
    SVLT.m                         SV + leverage + Student-t (§7.5c, reference baseline)
    SVLTRECH.m                     §8.2 PRIMARY — SRN cell
+priors/
    logPriorSVt.m
    logPriorSVLT.m
    logPriorSVLTRECH.m
+diagnostics/
    posteriorTrace.m               trace plots vs anneal step
    rHat.m                         R-hat across independent SMC runs
    mcseLogML.m                    Monte Carlo SE of log marginal likelihood
    relabelSignSwitch.m            post-process particles so beta_1 >= 0
scripts/
    run_stage1_simulation.m        the §11.1 simulation study
tests/
    tSimulatorRoundTrip.m          gate test — SV-LT-RECH recovers theta on synthetic data
config/experiments/
    stage1.yaml                    DGP θ_true, replicate count, SMC settings
    stage1_smoke.yaml              smaller variant for the CI test
```

### 4.2 Model specifications

#### 4.2.1 `+models/SVt.m` (Student-t innovations)

```
r_t = exp(h_t/2) * eps_t,                eps_t ~ t_nu / sqrt(nu/(nu-2))
h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t,    eta_t ~ N(0, 1)
```

`theta = [mu, phi, sigma_eta, nu]`, `nu > 2`.

Priors: as plain SV plus `nu ~ Gamma(1, 0.1)` (rate parameterisation, mean = 10).

The scaled-`t` normalisation `/sqrt(nu/(nu-2))` keeps `Var(eps_t) = 1` so `Var(r_t | h_t) = exp(h_t)` — important for comparability with plain SV.

`observationLogLik`: standard Student-t log-density (use `gammaln` for stability).

#### 4.2.2 `+models/SVLT.m` (leverage + Student-t)

Adds correlation `rho` between contemporaneous return shock and next-period log-variance innovation. **Cholesky coupling** (rejected the Omori–Chib–Shephard–Nakajima 2007 mixture for the first pass — switch to it if particles starve):

Generate `(z, eta_{t+1})` jointly:
```
[ z        ]   [ 1            0          ] [ u_1 ]
[ eta_{t+1}] = [ rho   sqrt(1-rho^2)     ] [ u_2 ]
u_1, u_2 ~ N(0, 1)
```

Then `eps_t` is built from `z` via the Student-t scale mixture: `eps_t = z / sqrt(W_t / nu)` with `W_t ~ chi^2_nu`.

`theta = [mu, phi, sigma_eta, rho, nu]`, `rho ∈ (-1, 1)`, `rho ~ Uniform(-1, 1)`.

`transitionSample` must condition on the previous-period return shock to produce `eta_{t+1}` correctly. This means the model has to track `epsilon_prev` between steps. Implementation contract:
- `transitionSample(hOld, theta, t, n, aux)` accepts an OPTIONAL 5th argument `aux` carrying `epsilonPrev`
- particle filter passes `aux` to make the proposal exact

Action: add an optional 5th argument to `Model.transitionSample` and update `+models/SV.m` to ignore it.

#### 4.2.3 `+models/SVLTRECH.m` — flagship (§8.2)

```
r_t  = exp(h_t/2) * eps_t,           eps_t ~ t_nu
h_t  = mu + phi (h_{t-1} - mu) + omega_t + sigma_eta * eta_t
omega_t = beta_0 + beta_1 * s_t
s_t  = ReLU( v^T x_t + w_h s_{t-1} + b ),    s_1 ≡ 0
x_t  = (h_{t-1}, r_{t-1}, omega_{t-1}, z_{t-1})^T
```

where `z_{t-1}` is the standardised covariate vector (may be empty).

`theta = [mu, phi, sigma_eta, rho, nu, beta_0, beta_1, v_h, v_r, v_omega, v_z(:), w_h, b]`.

Priors:
- SV core: as in SVLT
- `beta_0, beta_1 ~ Uniform(0, 0.5)` (signs are positive → β₀ on boundary is the label-switching risk documented in `docs/design_decisions.md` D7)
- `v_h, v_r, v_omega ~ N(0, 0.1)`
- `v_z` (covariate coefs): `N(0, 0.5)`
- `w_h, b ~ N(0, 0.1)`

The RNN cell is in `+cells/srn.m` (pure function). Calling convention:
```matlab
[s, ctx] = cells.srn(x, sPrev, weights)
```
where `weights = struct('v', v, 'w_h', w_h, 'b', b)` and `ctx` carries anything needed for the next call (nothing for SRN; LSTM and GRU need cell state).

### 4.3 SMC implications

When `SVLTRECH` is plugged into the existing engine, the bootstrap PF must:
1. Carry the running RNN hidden state `s_{t-1}` and prior `omega_{t-1}` per particle. Add a `particleAux` field to the filter's state.
2. Propagate `s_t` deterministically from `(s_{t-1}, x_t, theta)` via `+cells/srn.m`.
3. Sample `h_t` from the leverage-coupled Gaussian conditional on `(h_{t-1}, omega_t, eps_{t-1})`.

The cleanest way to extend the PF without forking `bootstrap.m`: let `Model` expose two extra methods:

```matlab
aux = initParticleAux(theta, n)
[aux, omega] = updateParticleAux(aux, h, y, theta, t)
```

Default implementations in `Model.m` return empty structs. `SVLTRECH` overrides them to carry `s_t` and `omega_t`. The PF then loops:

```matlab
% Inside +inference/+pf/bootstrap.m, replace single-line propagate with:
[aux, omega]      = model.updateParticleAux(aux, h, y, theta, t);
[h, epsilonPrev]  = model.transitionSample(h, theta, t, M, ...
                       struct('omega', omega, 'epsilonPrev', epsilonPrev));
```

This is the only non-trivial bootstrap.m change required for Phase 3.

### 4.4 Simulation-study protocol (`scripts/run_stage1_simulation.m`)

Mirrors NEU thesis §3.4.2.4 part a.

Inputs:
- `θ_true` from a known DGP. Suggested defaults (in `config/experiments/stage1.yaml`):
  - `mu = -0.5`
  - `phi = 0.97`
  - `sigma_eta = 0.25`
  - `rho = -0.5`
  - `nu = 8`
  - `beta_0 = 0.05`
  - `beta_1 = 0.20`
  - RNN weights drawn from prior, fixed across replicates
- `T = 2000` per replicate (matches §11.1)
- `nReplicates = 30`
- `nSeeds = 3` independent SMC seeds per replicate (for R-hat)

Per replicate:
1. Seed RNG = `baseSeed + 1000 * replicate`.
2. Simulate `(y, h)` of length `T` from `models.SVLTRECH.simulate(θ_true, T)`.
3. Fit via `inference.smc.likelihoodAnneal` with `N=2000`, `M=200`, `nSweeps=5`.
4. Across `nSeeds` parallel SMC fits, compute R-hat per parameter; flag any `R̂ > 1.1`.
5. Record `mean(theta_posterior)` and `std(theta_posterior)` per parameter.

Outputs (`results/stage1/`):
- `recovery.parquet` — long-form table: `replicate, seed, param, true, posteriorMean, posteriorStd, withinCI95`
- `mcse_logml.parquet` — MCSE of `log Z` across seeds, per replicate
- `figures/F11_stage1_recovery.png` — boxplot of `θ̂ − θ_true` per parameter
- `tables/T11_stage1_summary.csv` — for each parameter: mean recovery error, 95% CI coverage, MCSE

### 4.5 Go/No-Go criteria (the actual gate)

Phase 3 PASSES iff **all** the following hold across the 30-replicate study:

| Criterion | Target |
|---|---|
| Coverage of 95% credible interval | ≥ 90% per parameter |
| Median `\|posteriorMean − θ_true\|` | ≤ 1 posterior SD per parameter |
| R-hat across 3 seeds | ≤ 1.10 per parameter |
| MCSE of `log Z` across 3 seeds | ≤ 0.5 nats |
| `v_z` recovery when covariate has known zero effect | posterior CI must contain 0 in ≥ 90% of replicates |
| `v_z` recovery when covariate has known non-zero effect | posterior CI must NOT contain 0 in ≥ 80% of replicates |
| Sign-relabel diagnostic | < 5% of particles flipped post-fit |
| Particle-filter underflow guard | `clipLogWeight` triggers in < 1% of PF steps on average |

If ANY criterion fails, **STOP** and debug before any real-data work. The most likely fix points (architectural review surface area):

1. Bootstrap PF starves on Student-t tails → switch to `+inference/+pf/auxiliary.m` (auxiliary PF) — file scaffold required.
2. Leverage coupling biased → switch from Cholesky to Omori–Chib–Shephard–Nakajima (2007) mixture-of-normals representation.
3. RNN sign symmetry → `+diagnostics/relabelSignSwitch.m` clean-up + tighter `beta_1 ~ Uniform(eps, 0.5)` with `eps = 0.01`.

### 4.6 The simulator round-trip test contract

`tests/tSimulatorRoundTrip.m` is the CI form of the simulation study. Run it on a smaller config (`N=500, M=50, nReplicates=10, T=500`) so it fits in a CI minute; assert:

```matlab
testCase.verifyGreaterThanOrEqual(coverage_95, 0.80);   % weaker than paper gate
testCase.verifyLessThanOrEqual(rHatMax, 1.20);
```

Use `config/experiments/stage1_smoke.yaml` for the CI variant.

### 4.7 RNG protocol

All Phase 3 work uses `utils.reproducibility(baseSeed + replicate)` as the first line of every script. The base seed is `20260516`. Independent SMC seeds within a replicate use `baseSeed + 1000*replicate + seedIdx` so seeds never collide.

### 4.8 Recommended order of work

1. **`+models/SVt.m` + `+priors/logPriorSVt.m`** — extends SV with Student-t. Easiest; validates that the abstract Model interface scales. Add `tSvtRoundTrip` style test.
2. **`+models/SVLT.m`** — adds leverage. The Cholesky-coupled transition is the first place to break the SMC engine. Verify by simulator round-trip BEFORE moving on.
3. **`+cells/srn.m`** — pure function. Trivial; one test verifying numerical correctness.
4. **`+models/SVLTRECH.m`** — combines everything. Add `initParticleAux` / `updateParticleAux` overrides. Modify `bootstrap.m` to call them.
5. **`+diagnostics/{posteriorTrace, rHat, mcseLogML, relabelSignSwitch}.m`** — wire into `likelihoodAnneal` output.
6. **`scripts/run_stage1_simulation.m`** — 30-replicate full study.
7. Iterate on go/no-go failures until all criteria pass.

### 4.9 Things that are NOT Phase 3

- LSTM / GRU cells — those are Phase 4 (`+cells/lstm.m`, `+cells/gru.m`, `+models/SVLTRECHlstm.m`, `+models/SVLTRECHgru.m`).
- Auxiliary PF — write it only if bootstrap PF fails the underflow gate.
- Realized-volatility measurement equation (`RSVLTRECH`) — Phase 4.
- GARCH-family benchmarks — Phase 4.
- VB mean-field — Phase 4 optional path.
- MIDAS / LASSO extension — Phase 5 / Phase 6.

---

## 5. Quick-start for Phase 3

```bash
cd /home/d0/projects/finance_eng
source setup.sh

# Confirm Phase 1+2 still green
matlab -batch "addpath(pwd); pyenv('Version', '/home/d0/projects/finance_eng/.venv/bin/python'); runtests('tests')"
# All 30 tests must pass before starting Phase 3.

# Then build Phase 3 files in the order listed above, running tests after each.
matlab -batch "addpath(pwd); run('scripts/run_stage1_simulation.m')"
# Inspect results/stage1/ — gate via §4.5 criteria.
```

---

## 6. References

- Nguyen, Nguyen, Tran (2024). *Deep learning enhanced volatility modeling with covariates* (RECH-X). *Finance Research Letters* 70, 106145.
- NEU Hanoi thesis (2023). *Building Statistical Machine Learning Models for Volatility Forecasting*.
- Kim, Shephard, Chib (1998). *Stochastic Volatility: Likelihood Inference*. RES 65(3).
- Omori, Chib, Shephard, Nakajima (2007). *Stochastic volatility with leverage*. J. Econometrics 140(2).
- Hansen, Lunde, Nason (2011). *The model confidence set*. Econometrica 79(2).
- Politis, Romano (1994). *The stationary bootstrap*. JASA 89(428).
- Patton (2011). *Volatility forecast comparison using imperfect volatility proxies*. J. Econometrics 160(1).
- Liu, Wang, Tran, Kohn (2025). *RealRECH*. Economic Modelling 142, 106922.
- Duan, Fulop (2015). *Density-tempered marginalized SMC*. JAE 30(3).
- Nguyen, Tran, Gunawan, Kohn (2023). *SR-SV*. JBES 41(2).
