# SV-LT-RECH MATLAB Research Codebase — Implementation Plan

## Context

The project `finance_eng/` is a fresh, greenfield research repository (no source code yet) whose purpose is to **replicate and extend two prior papers** on hybrid econometric + deep-learning volatility forecasting:

1. Nguyen, Nguyen & Tran (2024). *Deep learning enhanced volatility modeling with covariates* (**RECH-X**). *Finance Research Letters* 70, 106145. — GARCH backbone with a Simple-RNN-augmented long-term variance + exogenous covariates.
2. NEU Hanoi thesis (2023). *Building Statistical Machine Learning Models for Volatility Forecasting*. — adds an LSTM-GARCH-MIDAS-LASSO contribution and a Vietnamese stock market application.

The user-authored design document `PROPOSED_METHODOLOGY.md` defines the **extension we are building**: replace the GARCH backbone with a **Stochastic Volatility (SV)** backbone, retain the RNN-augmented log-variance state, add **leverage** and **Student-t innovations**, and optionally couple a **realized-volatility measurement equation**. The flagship model is **SV-LT-RECH** with two co-primary RNN cells (SRN and LSTM); GRU is a sensitivity check; RSV-LT-RECH is the realized-data variant.

Both reference papers were read end-to-end this session. The two MATLAB codebases the original authors maintain (`VBayesLab/Stochastic-Volatility/lstmSV/` and `VBayesLab/RealRECH/SMC_for_RECH/`) provide ~80% of what we need at the algorithmic level but carry **no license file**, so we will clean-room reimplement in MATLAB rather than fork. `stochvol` (R, GPL-2) gives us a baseline SV-t-l calibration target we can shell out to from `tests/`.

User decisions (already confirmed):

- **Language:** MATLAB primary (mirrors the original lab's convention).
- **Scope:** everything in `PROPOSED_METHODOLOGY.md` — full §1–§16.
- **Data:** no data on disk; build the fetch pipeline.
- **Polish:** research repo (top-level scripts + `.mlx` notebooks, not a packaged toolkit). Clean code, minimal inline comments, `%%` section markers for narrative.

## Architectural decisions

**Three corrections to the naive "folder per model" layout** (surfaced by the architecture review):

1. Use a `+models/` MATLAB package with an abstract `Model` base class instead of copy-pasting four near-identical model folders. Each model overrides `logPrior`, `simulate`, `unpack`, `particleStep`. The four `*RECH*` variants share everything except the RNN cell.
2. The SMC engine receives a `Model` *handle object* (not a closure / function handle). Closures in MATLAB capture workspace copies which silently break under `parfor`; handles propagate cleanly and are `save`/`load`-friendly.
3. Split inference into siblings — `+inference/+smc`, `+inference/+pf`, `+inference/+mcmc`, `+inference/+vb` — so the Variational Bayes alternative (§9.5) and the proximal-LASSO MIDAS step (§11.5) have first-class homes rather than being grafted on later.

**Numerical safety from line 1** (the single biggest failure mode in this model family):

- Bootstrap particle filter operates in **log-weight space** with `logsumexp`. Never store raw weights.
- Leverage `corr(εₜ, ηₜ₊₁) = ρ` makes the one-step-ahead state proposal non-Gaussian; we implement the Omori-Chib-Shephard-Nakajima (2007) mixture-of-normals representation OR a Cholesky-coupled proposal. Decision: start with **Cholesky coupling** (simpler, valid for any ρ ∈ (−1, 1)); switch to mixture if particles degenerate.
- Auxiliary particle filter (`+inference/+pf/auxiliary.m`) is available behind the same interface as the bootstrap PF; we keep both because Student-t tails with ν ≈ 4 routinely starve the bootstrap PF.
- Sign-relabeling for ReLU SRN cells: after each SMC run, post-process particles so β₁ ≥ 0 (since the prior already constrains β₁ ≥ 0, this is a *check*, not a fix — but we add it to catch boundary-mode collapse). For the LSTM cell, no relabeling is needed (gates break the sign symmetry).

## Repository structure

```
finance_eng/                              (root, already exists)
├── PROPOSED_METHODOLOGY.md               existing — single source of truth
├── methodlogy_ref/                       existing — OCR'd reference papers
├── README.md                             new — install + reproduce instructions
├── LICENSE                               new — MIT
├── docs/
│   ├── data_sources.md                   how each dataset is fetched + license/citation
│   ├── reproducibility_checklist.md      pinned MATLAB version, toolboxes, seeds
│   └── design_decisions.md               record of every non-obvious choice
├── +models/                              ABSTRACT BASE + concrete model classes
│   ├── Model.m                           abstract — logPrior, simulate, unpack, particleStep
│   ├── GARCH.m                           §7.1
│   ├── GJRGARCH.m                        §7.2
│   ├── GARCHX.m                          §7.3
│   ├── RealGARCH.m                       §7.4
│   ├── SV.m                              §7.5(a) plain SV — calibration target
│   ├── SVt.m                             §7.5(b)
│   ├── SVLT.m                            §7.5(c) SV + leverage + t (reference baseline)
│   ├── SVLTRECH.m                        §8.2 PRIMARY — SRN cell
│   ├── SVLTRECHlstm.m                    §8.4 CO-PRIMARY — LSTM cell
│   ├── SVLTRECHgru.m                     §8.5 sensitivity — GRU cell
│   └── RSVLTRECH.m                       §8.3 realized-vol measurement variant
├── +cells/                               PURE FUNCTIONS — RNN cell forward passes
│   ├── srn.m                             ReLU simple-RNN step
│   ├── lstm.m                            LSTM cell step
│   └── gru.m                             GRU cell step
├── +inference/
│   ├── +smc/
│   │   ├── likelihoodAnneal.m            §9.2 in-sample
│   │   ├── dataAnneal.m                  §9.4 rolling OOS
│   │   ├── adaptiveTemperature.m         bisect aₖ so cond. ESS = N/2
│   │   └── ess.m
│   ├── +pf/
│   │   ├── bootstrap.m                   bootstrap PF (log-weight space)
│   │   ├── auxiliary.m                   auxiliary PF for heavy-tailed cases
│   │   ├── resampleSystematic.m
│   │   └── resampleStratified.m
│   ├── +mcmc/
│   │   ├── rwMetropolis.m                random-walk Metropolis sweep
│   │   ├── adaptiveMetropolis.m          covariance-adapted proposal
│   │   └── proximalStep.m                §11.5 LASSO soft-threshold inside MCMC
│   └── +vb/
│       ├── meanFieldGaussian.m           §9.5 fast prototyping alternative
│       └── elbo.m
├── +diagnostics/
│   ├── posteriorTrace.m                  trace plots
│   ├── rHat.m                            R̂ across independent SMC runs
│   ├── mcseLogML.m                       Monte Carlo SE of log marginal likelihood
│   ├── relabelSignSwitch.m               post-process SRN weight signs
│   ├── essCurve.m                        ESS vs anneal temperature
│   └── residualChecks.m                  Ljung-Box / QQ / kurtosis on ε̂
├── +data/
│   ├── +fetch/
│   │   ├── manifest.yaml                 pinned URLs + checksums per dataset
│   │   ├── cacheFetch.m                  HTTP wrapper that hits local cache first
│   │   ├── fetchYahoo.m                  daily prices via Yahoo CSV endpoint
│   │   ├── fetchRealizedLibrary.m        OMI archive + Liu et al. 2025 extended panel
│   │   ├── fetchFRED.m                   FRED CSV API
│   │   ├── fetchInvesting.m              investing.com (5-market panel)
│   │   └── fetchVN.m                     vn.investing.com + vnstock (Python interop)
│   └── +preprocess/
│       ├── stationarityTests.m           ADF + PP + KPSS
│       ├── meanEquation.m                μ = 0 default; ARMA(p,q) by BIC fallback
│       ├── residualDiagnostics.m         Ljung-Box, McLeod-Li, ARCH-LM, JB, BDS
│       ├── standardize.m                 in-sample stats only — leakage-safe
│       └── trainTestSplit.m              chronological split
├── +eval/                                §10 forecast evaluation
│   ├── pps.m  quantileScore.m  mseMae.m  r2log.m  qlike.m
│   ├── dieboldMariano.m                  HAC-robust DM test
│   └── modelConfidenceSet.m              Hansen-Lunde-Nason 2011
├── +priors/                              §9.1
│   ├── logPriorSV.m
│   ├── logPriorGARCH.m
│   └── logPriorRECH.m
├── +utils/
│   ├── reproducibility.m                 rng seed setup
│   ├── saveResults.m                     write posteriors/forecasts/scores to disk
│   ├── loadConfig.m                      YAML/JSON experiment config loader
│   ├── parPool.m                         managed parpool startup
│   ├── logsumexp.m  log1pexp.m           numerical primitives
│   └── plotForecast.m
├── scripts/                              REPRODUCIBILITY entry points
│   ├── make_all.m                        top-level orchestrator
│   ├── run_phase1_smc_sanity.m           plain SV vs R stochvol cross-check
│   ├── run_stage1_simulation.m           §11.1 — 30 replicates, identifiability sweep
│   ├── run_stage2_oxford_man.m           §11.2 — 10 markets with RV
│   ├── run_stage3_covariates.m           §11.3 — 5 markets with VIX/OIL/GOLD/EXR
│   ├── run_stage4_vietnam.m              §11.4 — VN-Index + HNX-Index
│   ├── run_stage5_midas.m                §11.5 — optional MIDAS-LASSO extension
│   ├── calibrate_against_stochvol.m      R bridge for baseline SV calibration
│   └── make_paper_tables.m               T1–T7 + F1–F4
├── notebooks/                            MATLAB Live Scripts (.mlx)
│   ├── 01_data_exploration.mlx
│   ├── 02_baselines_walkthrough.mlx
│   └── 03_proposed_models_results.mlx
├── config/                               YAML configs per experiment
│   ├── smc_defaults.yaml                 N=5000, M=200, ESS=0.5N, 10 RWM moves
│   ├── priors.yaml                       all priors from §9.1
│   ├── markets.yaml                      tickers + date ranges per market
│   └── experiments/
│       ├── stage1.yaml  stage2.yaml  stage3.yaml  stage4.yaml  stage5.yaml
├── tests/                                MATLAB built-in unittest framework
│   ├── tParticleVsGrid.m                 PF log-lik vs brute-force grid (±0.5 nats)
│   ├── tResamplingInvariance.m           PF log-lik MC SE scales as 1/√M
│   ├── tEssGradient.m                    ESS monotone in (aₖ − aₖ₋₁)
│   ├── tPriorPredictive.m                no-data SMC recovers prior
│   ├── tSimulatorRoundTrip.m             §11.1 in CI form
│   ├── tCrossEngineSV.m                  plain SV vs R stochvol — ±2σ agreement
│   ├── tRNNCells.m                       SRN/LSTM/GRU numerical correctness
│   ├── tEvalMetrics.m                    PPS/QS/MSE/MAE/R²LOG on known inputs
│   └── tPriorsDensity.m                  prior log-density matches pdf hand calc
├── data/                                 gitignored
│   ├── raw/                              cache of pulled files (keyed by manifest)
│   └── processed/                        .parquet of preprocessed (market, freq, split)
└── results/                              gitignored
    ├── posteriors/  forecasts/  scores/  figures/  tables/
```

## Implementation phasing (6 weeks)

Each phase ends with a runnable artifact and at least one CI-form test.

**Phase 1 — SMC engine + plain SV sanity (Week 1)**
- Implement `+models/Model.m`, `+models/SV.m`, `+priors/logPriorSV.m`.
- Implement `+inference/+pf/bootstrap.m` (log-space, no leverage yet), `+inference/+smc/likelihoodAnneal.m`, `+inference/+smc/adaptiveTemperature.m`, `+inference/+smc/ess.m`.
- Implement `+utils/logsumexp.m`, `+utils/reproducibility.m`.
- Bridge script `scripts/calibrate_against_stochvol.m` shells out to R, fits `stochvol::svsample`, compares posterior means within 2σ.
- Tests passing: `tCrossEngineSV.m`, `tParticleVsGrid.m`, `tResamplingInvariance.m`, `tEssGradient.m`, `tPriorPredictive.m`, `tPriorsDensity.m`.

**Phase 2 — data + diagnostics (Week 1, parallel with Phase 1)**
- `+data/+fetch/manifest.yaml` lists every dataset URL with SHA-256.
- `+data/+fetch/cacheFetch.m` reads cached if checksum matches; else downloads.
- All four fetchers (`fetchYahoo`, `fetchRealizedLibrary`, `fetchFRED`, `fetchVN`) — `fetchVN` uses Python interop to call `vnstock` since investing.com has no API.
- Full `+data/+preprocess/` suite.
- `notebooks/01_data_exploration.mlx` produces Table T1 (descriptives + ADF/JB/ARCH-LM/BDS per series) for all 12 markets.
- Tests passing: `tEvalMetrics.m`, manifest checksum verification.

**Phase 3 — simulation study + identifiability (Week 2)** ← *moved earlier than original plan*
- Implement `+models/SVLT.m` and `+models/SVLTRECH.m` (SRN cell, leverage via Cholesky coupling).
- Implement `+cells/srn.m`, `+inference/+mcmc/rwMetropolis.m`.
- `scripts/run_stage1_simulation.m` — 30 replicate runs from known θ; identifiability sweep across each parameter.
- Gate: SV-LT-RECH recovers θ within 2 posterior SD on ≥27/30 replicates. If not, **stop and debug** before any real-data fit.
- Add `+diagnostics/posteriorTrace.m`, `rHat.m`, `mcseLogML.m`, `relabelSignSwitch.m`.
- Test added: `tSimulatorRoundTrip.m`.

**Phase 4 — baselines + remaining proposed models (Week 3)**
- Implement `+models/SVt.m`, `+models/GARCH.m`, `+models/GJRGARCH.m`, `+models/GARCHX.m`, `+models/RealGARCH.m`.
- Implement `+models/SVLTRECHlstm.m`, `+models/SVLTRECHgru.m`, `+models/RSVLTRECH.m`, `+cells/lstm.m`, `+cells/gru.m`.
- Implement `+inference/+pf/auxiliary.m` (fallback for heavy-tail particle starvation).
- Add `+inference/+vb/meanFieldGaussian.m` for fast prototyping iterations.

**Phase 5 — empirical applications (Week 4)**
- `run_stage2_oxford_man.m`: 10 markets, RSV-LT-RECH vs 7 baselines. Use Liu et al. 2025 RealRECH archived RV panel as the OMI replacement (OMI was decommissioned 2022).
- `run_stage3_covariates.m`: 5 markets (VN100, N225, CAC40, ASX, BVSP) with VIX/OIL/GOLD/EXR covariates.
- `run_stage4_vietnam.m`: VN-Index + HNX-Index with crude oil / USD-VND / BTC / gold covariates. 2,262 obs, 1,500 train / 762 test (matches NEU thesis §3.5).
- DM tests and MCS run on QLIKE loss.

**Phase 6 — tables, figures, robustness (Weeks 5–6)**
- `make_paper_tables.m` produces T1–T7 + F1–F4 from saved `results/`.
- Robustness: GRU variant, alternative priors, jump-augmented variant if time allows.
- Optional Stage 5 — `run_stage5_midas.m` with proximal-LASSO inside SMC (`+inference/+mcmc/proximalStep.m`).
- Paper draft.

## Reuse opportunities (clean-room reimplementation in MATLAB)

| Source | What we mirror | Where it lands |
|--------|-----------------|----------------|
| `VBayesLab/Stochastic-Volatility/lstmSV/` | Particle filter step structure, simulator pattern, model-as-object idiom | `+inference/+pf/bootstrap.m`, `+models/SVLTRECHlstm.m`, `+models/SV.m` |
| `VBayesLab/RealRECH/SMC_for_RECH/` (`RECH_LikAnneal.m`, `RECH_DataAnneal.m`) | Likelihood-annealing schedule, Markov-move loop, ESS adaptation | `+inference/+smc/likelihoodAnneal.m`, `+inference/+smc/dataAnneal.m`, `+inference/+smc/adaptiveTemperature.m` |
| `numpyro` SV example (Apache-2.0) | Idiomatic SV model spec, Student-t parameterization | `+models/SV.m`, `+models/SVt.m` |
| `stochvol` R package (GPL-2 — used as **external calibration target**, no code copied) | SV-t-l posterior to compare against | `scripts/calibrate_against_stochvol.m`, `tests/tCrossEngineSV.m` |

## Risks and mitigations

1. **Oxford-Man Realized Library decommissioned 2022.** Use Liu et al. 2025 RealRECH archived 31-index panel as the drop-in; pin a Wayback snapshot of OMI as a secondary source. Documented in `docs/data_sources.md`.
2. **Particle-filter underflow with Student-t × leverage × RNN.** Log-weight space from line 1; auxiliary PF behind the same interface; particle-weight clamping with a diagnostic counter; Cholesky-coupled state proposal under leverage.
3. **Label-switching of RNN weight signs.** β₀ ≥ 0, β₁ ≥ 0 prior caps one symmetry; `relabelSignSwitch.m` post-process catches the rest; R̂ across independent SMC runs flags inter-mode disagreement that MCSE alone misses.
4. **Vietnam data has no public API.** `fetchVN.m` calls Python's `vnstock` via MATLAB-Python interop. Cached locally so it runs once; manifest pins the snapshot date.
5. **`parfor` foot-guns.** Models are *handle objects*, not closures; `+utils/parPool.m` enforces a managed pool with explicit worker count from config.

## Critical files to create (priority order)

1. `+models/Model.m` — abstract base class; gates every concrete model.
2. `+inference/+pf/bootstrap.m` — log-space particle filter; primary numerical risk.
3. `+inference/+smc/likelihoodAnneal.m` — main estimator.
4. `+models/SV.m` + `scripts/calibrate_against_stochvol.m` — first end-to-end sanity check.
5. `tests/tParticleVsGrid.m` — cheapest test that catches the worst class of bugs.
6. `+models/SVLTRECH.m` + `+cells/srn.m` — flagship model.
7. `+data/+fetch/manifest.yaml` + `cacheFetch.m` — locks all external sources.
8. `scripts/run_stage1_simulation.m` — go/no-go gate for real-data work.

## Verification plan

End-to-end verification rests on six checks, ordered cheapest-first:

1. **Prior log-density unit tests** — closed-form pdf vs `logPrior` for each parameter.
2. **Particle filter vs brute-force grid filter** — at fixed θ on a synthetic series, PF log-lik agrees with a 1,000-point grid filter on h ∈ [−10, 10] to ±0.5 nats.
3. **Resampling invariance** — PF log-lik mean over 50 replicates at fixed θ has SE ∝ 1/√M.
4. **Prior-predictive recovery** — SMC on no-data target should recover the prior (catches likelihood-sign bugs).
5. **Cross-engine plain-SV calibration** — `+models/SV.m` posterior agrees with R `stochvol::svsample` to ±2σ on a 1,000-day SPX sample.
6. **Simulator round-trip on SV-LT-RECH** — 30 synthetic series, parameters recovered within 2 posterior SD on ≥90% of replicates.

Empirical reproduction targets (must match published numbers within MCSE):

- S&P 500 Mar.llh: GARCH ≈ −1556, RealGARCH ≈ −1595, RECH-X ≈ −1487 (RECH-X paper Table 1).
- N225 RECH-X v_RV posterior mean ≈ 0.617 ± 0.186 (RECH-X paper Table 4).
- VN-Index descriptives: mean ≈ 0.0149, std ≈ 0.498, skew ≈ −0.987, kurtosis ≈ 7.73 (NEU thesis Table 4.14).

If our SV-LT-RECH log marginal likelihood does NOT beat RECH-X on the same datasets, the experiment has failed at the methodology level (not the implementation), and we report that honestly rather than re-tune until it wins.
