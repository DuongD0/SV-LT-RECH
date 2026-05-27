# Phase 5 — Stage-4 Vietnam Vertical Slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full data→fit→forecast→score pipeline end-to-end on the VN-Index with four covariates, comparing the proposed SV-LT-RECH (SRN) against GARCH-t, GJR-t, and SVLT baselines on five predictive scores.

**Architecture:** Extend the bootstrap PF to expose per-step predictive densities + variances; add a fixed-parameter rolling-predictive routine that Bayesian-model-averages those over a thinned posterior; wire a Stage-4 pipeline (`+experiments/runStage4.m`) that the network-facing `scripts/run_stage4_vietnam.m` calls after loading data. Cheap models (GARCH) reuse `garch.rollingForecast`; SV/RECH use the new rolling predictive. See `docs/PHASE_5_PLAN.md` for the design spec.

**Tech Stack:** MATLAB R2026a (Econometrics + Statistics + Parallel toolboxes), `matlab.unittest`. Python venv (vnstock/FRED fetch) via `pyenv`. Reference: `PROPOSED_METHODOLOGY.md` §9.4, §10.

---

## Conventions for this plan

- **MATLAB binary** is not on `PATH`: invoke `/home/d0/MATLAB/R2026a/bin/matlab`.
- **Run the whole suite:**
  `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); runtests('tests')"`
- **Run one test file:**
  `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tNAME.m')"`
  (Do **not** use `runtests({'tests/tNAME'})` without `.m` — it fails to build a suite. Memory: `matlab-env`.)
- **First MATLAB launch after a reboot can hang** on MathWorks service init; if a `-batch` produces zero output for minutes, kill the process tree and relaunch. Probe health with `-batch "disp(1+1)"`.
- **This repo is not git-initialized.** "Checkpoint" steps run the relevant tests instead of committing. If you want version control, run `git init` before starting (optional, not required).
- **Source the env first** in any shell that calls MATLAB: `source setup.sh` (activates venv + puts toolboxes in reach).

---

## File structure (locked before tasks)

| File | Responsibility |
|---|---|
| `+inference/+pf/bootstrap.m` (modify) | additive `opts.returnIncrements` → also emit `T×1` per-step predictive log-density + one-step predicted variance |
| `+inference/+forecast/rollingPredictive.m` (create) | fixed-parameter rolling 1-step-ahead Bayesian predictive over a thinned posterior |
| `+experiments/runStage4.m` (create) | pure pipeline: `(y, Z, splitIdx, cfg)` → fit 4 models → forecast → score → DM + MCS → results struct (+ optional CSV/figure) |
| `+data/loadStage4.m` (create) | network data loader: VN-Index + 4 covariates → aligned `(y, Z, splitIdx, dates)` |
| `scripts/run_stage4_vietnam.m` (create) | thin entry point: load config + data, call `experiments.runStage4`, write `results/stage4/` |
| `config/experiments/stage4.yaml` (create) | market, dates, split, SMC sizes, `J`, QS levels, seed |
| `tests/tBootstrapIncrements.m` (create) | per-step increments sum to scalar logLik; variance path positive |
| `tests/tRollingPredictive.m` (create) | shapes, finiteness, BMA bounded by per-particle range, `J=1` reduces to single PF |
| `tests/tStage4Smoke.m` (create) | network-free end-to-end on synthetic fixture; `scores` well-formed for all 4 models |

---

### Task 1: Expose per-step predictive density + variance from the bootstrap PF

**Files:**
- Modify: `+inference/+pf/bootstrap.m`
- Test: `tests/tBootstrapIncrements.m`

- [ ] **Step 1: Write the failing test**

Create `tests/tBootstrapIncrements.m`:

```matlab
classdef tBootstrapIncrements < matlab.unittest.TestCase
% tBootstrapIncrements  PF per-step predictive density + variance outputs.

    methods (Test)

        function incrementsSumToScalarLogLik(testCase)
            rng(2026, 'threefry');
            mdl   = models.SV();
            theta = [-0.2, 0.95, 0.30];                 % [mu, phi, sigma_eta]
            [y, ~] = mdl.simulate(theta, 300);

            rng(7, 'threefry');
            logLik = inference.pf.bootstrap(mdl, y, theta, struct('M', 400));

            rng(7, 'threefry');                          % same PF randomness
            [logLik2, ~, lpd, vf] = inference.pf.bootstrap( ...
                mdl, y, theta, struct('M', 400, 'returnIncrements', true));

            testCase.verifyEqual(logLik2, logLik, 'RelTol', 1e-10);
            testCase.verifyEqual(numel(lpd), numel(y));
            testCase.verifyEqual(sum(lpd), logLik2, 'RelTol', 1e-10);
            testCase.verifySize(vf, [numel(y), 1]);
            testCase.verifyTrue(all(vf > 0));
            testCase.verifyTrue(all(isfinite(lpd)));
        end

        function defaultCallUnchanged(testCase)
            rng(3, 'threefry');
            mdl   = models.SV();
            theta = [0.0, 0.9, 0.4];
            [y, ~] = mdl.simulate(theta, 200);
            % Old 2-output contract still works.
            [logLik, hF] = inference.pf.bootstrap(mdl, y, theta, ...
                struct('M', 200, 'returnPath', true));
            testCase.verifyTrue(isscalar(logLik) && isfinite(logLik));
            testCase.verifySize(hF, [numel(y), 1]);
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tBootstrapIncrements.m')"`
Expected: FAIL — `incrementsSumToScalarLogLik` errors (too many output arguments) because `bootstrap` returns only `[logLik, hFiltered]`.

- [ ] **Step 3: Modify `bootstrap.m` — signature + defaults**

Change the function line (currently `function [logLik, hFiltered] = bootstrap(model, y, theta, opts)`) to:

```matlab
function [logLik, hFiltered, logPredDensity, varForecast] = bootstrap(model, y, theta, opts)
```

In the `defaults` block (after `defaults.returnPath = false;`) add:

```matlab
    defaults.returnIncrements = false;   % also emit per-step pred density + variance
```

After the `if opts.returnPath ... else hFiltered = []; end` block, add preallocation:

```matlab
    if opts.returnIncrements
        logPredDensity = zeros(T, 1);
        varForecast    = zeros(T, 1);
    else
        logPredDensity = [];
        varForecast    = [];
    end
```

- [ ] **Step 4: Modify `bootstrap.m` — capture inside the loop**

Inside the `for t = 1:T` loop, immediately AFTER the line
`logLik = logLik + logLikIncrement;` and BEFORE `logW = logW + logW_inc - logLikIncrement;`, insert:

```matlab
        % One-step-ahead predictive density and predicted variance, using the
        % pre-update (t-1) normalised weights and the propagated h_t draws.
        % Var(y_t | h_t) = exp(h_t) for the scaled-t / Gaussian observation,
        % so E[exp(h_t) | y_{1:t-1}] is the variance forecast.
        if opts.returnIncrements
            logPredDensity(t) = logLikIncrement;
            varForecast(t)    = exp(utils.logsumexp(logW + h));
        end
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tBootstrapIncrements.m')"`
Expected: PASS (2/2).

- [ ] **Step 6: Checkpoint — regression-guard the PF callers**

Run the existing PF/SMC tests to confirm the additive change broke nothing:
`/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tParticleVsGrid.m'); runtests('tests/tParticleAuxDefault.m')"`
Expected: PASS for both.

---

### Task 2: Fixed-parameter rolling Bayesian predictive

**Files:**
- Create: `+inference/+forecast/rollingPredictive.m`
- Test: `tests/tRollingPredictive.m`

- [ ] **Step 1: Write the failing test**

Create `tests/tRollingPredictive.m`:

```matlab
classdef tRollingPredictive < matlab.unittest.TestCase
% tRollingPredictive  Fixed-parameter rolling 1-step-ahead predictive.

    methods (Test)

        function shapesAndPositivity(testCase)
            rng(11, 'threefry');
            mdl    = models.SV();
            theta  = [-0.2, 0.95, 0.30];
            [y, ~] = mdl.simulate(theta, 400);
            splitIdx = 300;

            % A small "posterior": jittered copies of the truth.
            N = 40;
            thetaPost = theta + 0.02 * randn(N, 3);
            thetaPost(:,2) = min(max(thetaPost(:,2), -0.99), 0.99);   % phi in (-1,1)
            thetaPost(:,3) = abs(thetaPost(:,3));                     % sigma_eta > 0

            out = inference.forecast.rollingPredictive(mdl, y, thetaPost, ...
                splitIdx, struct('J', 20, 'M', 200));

            nTest = numel(y) - splitIdx;
            testCase.verifySize(out.logPredDensity, [nTest, 1]);
            testCase.verifySize(out.varForecast,    [nTest, 1]);
            testCase.verifyEqual(out.testIdx, (splitIdx+1:numel(y))');
            testCase.verifyTrue(all(isfinite(out.logPredDensity)));
            testCase.verifyTrue(all(out.varForecast > 0));
        end

        function bmaWithinPerParticleRange(testCase)
            % log-mean-exp over particles must lie within [min, max] of the
            % per-particle predictive log-densities at every t.
            rng(12, 'threefry');
            mdl    = models.SV();
            theta  = [0.0, 0.92, 0.35];
            [y, ~] = mdl.simulate(theta, 250);
            splitIdx = 180;
            thetaPost = repmat(theta, 8, 1) + 0.01 * randn(8, 3);
            thetaPost(:,3) = abs(thetaPost(:,3));

            out = inference.forecast.rollingPredictive(mdl, y, thetaPost, ...
                splitIdx, struct('J', 8, 'M', 200, 'returnPerParticle', true));

            lo  = min(out.logPredPerParticle, [], 2);
            hi  = max(out.logPredPerParticle, [], 2);
            bma = out.logPredDensityAll;     % full-series BMA path
            testCase.verifyTrue(all(bma >= lo - 1e-9));
            testCase.verifyTrue(all(bma <= hi + 1e-9));
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tRollingPredictive.m')"`
Expected: FAIL — `inference.forecast.rollingPredictive` does not exist (unrecognised function/package).

- [ ] **Step 3: Create the package directory and function**

Create `+inference/+forecast/rollingPredictive.m`:

```matlab
function out = rollingPredictive(model, y, thetaPost, splitIdx, opts)
% rollingPredictive  Fixed-parameter rolling 1-step-ahead Bayesian predictive.
%
%   out = rollingPredictive(model, y, thetaPost, splitIdx, opts)
%
%   Holds the in-sample posterior fixed (PROPOSED_METHODOLOGY.md §9.4 sanity
%   option) and rolls a bootstrap PF forward over the full series for each of
%   J thinned posterior particles, then Bayesian-model-averages the per-step
%   predictive densities in log space and the predicted variances linearly.
%
%   Inputs
%   ------
%     model     : a models.Model handle. For covariate models (SVLTRECH) the
%                 caller MUST bind the FULL-series covariates via
%                 model.setCovariates(Z_full) before calling.
%     y         : (T_total)-by-1 full return series (train + test).
%     thetaPost : N-by-K in-sample posterior particles (from likelihoodAnneal).
%     splitIdx  : last in-sample index; test window is splitIdx+1 .. T_total.
%     opts      : struct
%                   .J                 thinned-particle count (default 200)
%                   .M                 PF latent particles (default 200)
%                   .pfOpts            extra opts forwarded to bootstrap
%                   .returnPerParticle keep the J-column matrices (default false)
%
%   Output (struct `out`)
%   ---------------------
%     .testIdx            test-window indices (splitIdx+1 .. T_total)'
%     .logPredDensity     test-window BMA log predictive density
%     .varForecast        test-window BMA one-step predicted variance
%     .logPredDensityAll  full-series BMA log predictive density
%     .varForecastAll     full-series BMA predicted variance
%     .logPredPerParticle (T_total-by-J) only if opts.returnPerParticle

    arguments
        model               models.Model
        y           (:,1)   double
        thetaPost   (:,:)   double
        splitIdx    (1,1)   double {mustBeInteger, mustBePositive}
        opts                struct = struct()
    end

    defaults.J                 = 200;
    defaults.M                 = 200;
    defaults.pfOpts            = struct();
    defaults.returnPerParticle = false;
    opts = mergeStruct(defaults, opts);

    Ttot = numel(y);
    assert(splitIdx < Ttot, 'rollingPredictive:badSplit', ...
        'splitIdx %d must be < series length %d', splitIdx, Ttot);

    N = size(thetaPost, 1);
    J = min(opts.J, N);
    % Even thinning across the posterior particle order (particles are
    % equally weighted after the final anneal step).
    pick      = unique(round(linspace(1, N, J)));
    thetaThin = thetaPost(pick, :);
    J         = size(thetaThin, 1);

    pfOpts                  = mergeStruct(struct('M', opts.M), opts.pfOpts);
    pfOpts.returnIncrements = true;

    logPredAll = zeros(Ttot, J);
    varAll     = zeros(Ttot, J);
    for j = 1:J
        [~, ~, lpd, vf] = inference.pf.bootstrap(model, y, thetaThin(j, :), pfOpts);
        logPredAll(:, j) = lpd;
        varAll(:, j)     = vf;
    end

    % Bayesian model average: log-mean-exp across particles (rowwise, stable).
    m          = max(logPredAll, [], 2);
    logPredBMA = m + log(sum(exp(logPredAll - m), 2)) - log(J);
    varBMA     = mean(varAll, 2);

    testIdx               = (splitIdx + 1 : Ttot)';
    out.testIdx           = testIdx;
    out.logPredDensity    = logPredBMA(testIdx);
    out.varForecast       = varBMA(testIdx);
    out.logPredDensityAll = logPredBMA;
    out.varForecastAll    = varBMA;
    if opts.returnPerParticle
        out.logPredPerParticle = logPredAll;
    end
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tRollingPredictive.m')"`
Expected: PASS (2/2).

- [ ] **Step 5: Checkpoint**

Re-run Task-1 + Task-2 tests together:
`/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tBootstrapIncrements.m'); runtests('tests/tRollingPredictive.m')"`
Expected: PASS (4 total).

---

### Task 3: Stage-4 experiment config

**Files:**
- Create: `config/experiments/stage4.yaml`

- [ ] **Step 1: Create the config**

Create `config/experiments/stage4.yaml`:

```yaml
# Stage 4 — Vietnam VN-Index empirical application (PROPOSED_METHODOLOGY §11.4).
# Fixed-parameter rolling OOS (docs/PHASE_5_PLAN.md §3). Run locally:
#   matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); run('scripts/run_stage4_vietnam.m')"

market: VNINDEX

data:
  start:        '2016-01-04'
  end:          '2025-01-31'
  trainEnd:     1500            # 1500 train / rest test (NEU thesis §3.5.3)
  covariates:   [oil, usdvnd, btc, gold]

model:
  nCovariates: 4
  leverage:    cholesky

smc:
  N:             2000
  M:             200
  nSweeps:       5
  proposalScale: 0.5
  essThreshold:  0.5
  targetEss:     0.5
  verbose:       false

forecast:
  J: 200                        # thinned posterior particles for rolling predictive

eval:
  alphaQS: [0.01, 0.05]         # VaR levels for the quantile score
  mcsB:    2000                 # MCS bootstrap replicates

particleFilter:
  clipLogWeight: -50

baseSeed: 20260516
```

- [ ] **Step 2: Verify it loads + merges with defaults**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); c = utils.loadConfig('config/smc_defaults.yaml','config/experiments/stage4.yaml'); disp(c.smc.N); disp(c.model.nCovariates); disp(c.data.trainEnd)"`
Expected output includes `2000`, `4`, `1500`.

- [ ] **Step 3: Checkpoint** — config loads; no test file needed (exercised by Task 5 smoke).

---

### Task 4: Stage-4 pipeline core (`experiments.runStage4`)

**Files:**
- Create: `+experiments/runStage4.m`
- Test: `tests/tStage4Smoke.m`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/tStage4Smoke.m`:

```matlab
classdef tStage4Smoke < matlab.unittest.TestCase
% tStage4Smoke  Network-free end-to-end Stage-4 pipeline on a synthetic fixture.

    methods (TestMethodSetup)
        function requireEcon(testCase)
            testCase.assumeTrue(~isempty(ver('econ')), ...
                'Econometrics Toolbox not available; skipping Stage-4 smoke.');
        end
    end

    methods (Test)
        function pipelineProducesScoresForAllModels(testCase)
            rng(99, 'threefry');

            % Synthetic VN-like series from the flagship DGP with 2 covariates.
            K        = 2;
            T        = 280;
            splitIdx = 210;
            thetaTrue = [-0.5, 0.95, 0.25, -0.4, 8, 0.05, 0.20, ...
                         0.05, -0.05, 0.05, 0.10, -0.10, 0.10, 0.0];  % 12 + 2
            Z   = randn(T, K);
            gen = models.SVLTRECH('nCovariates', K, 'leverage', 'cholesky');
            gen.setCovariates(Z);
            [y, ~] = gen.simulate(thetaTrue, T);

            cfg = struct();
            cfg.model    = struct('nCovariates', K, 'leverage', 'cholesky');
            cfg.smc      = struct('N', 250, 'M', 40, 'nSweeps', 2, ...
                                  'proposalScale', 0.5, 'essThreshold', 0.5, ...
                                  'targetEss', 0.5, 'verbose', false);
            cfg.forecast = struct('J', 20);
            cfg.eval     = struct('alphaQS', [0.01, 0.05], 'mcsB', 200);
            cfg.particleFilter = struct('clipLogWeight', -50);
            cfg.baseSeed = 20260516;

            out = experiments.runStage4(y, Z, splitIdx, cfg);

            expected = {'GARCH-t','GJR-t','SVLT','SVLTRECH'};
            testCase.verifyEqual(out.modelNames, expected);

            S = out.scores;     % struct array, one per model
            testCase.verifyEqual(numel(S), 4);
            for m = 1:numel(S)
                testCase.verifyTrue(isfinite(S(m).pps));
                testCase.verifyTrue(isfinite(S(m).qlike));
                testCase.verifyTrue(isfinite(S(m).mse));
                testCase.verifyTrue(isfinite(S(m).mae));
                testCase.verifyTrue(isfinite(S(m).r2log));
                testCase.verifyTrue(isfinite(S(m).qs1));
                testCase.verifyTrue(isfinite(S(m).qs5));
            end

            % MCS keeps at least one model; DM table has 3 pairwise rows.
            testCase.verifyGreaterThanOrEqual(sum(out.mcs.inSet), 1);
            testCase.verifyEqual(numel(out.dmVsProposed), 3);
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tStage4Smoke.m')"`
Expected: FAIL — `experiments.runStage4` does not exist.

- [ ] **Step 3: Create `+experiments/runStage4.m`**

```matlab
function out = runStage4(y, Z, splitIdx, cfg, opts)
% runStage4  Stage-4 empirical pipeline: fit, OOS-forecast, score, compare.
%
%   out = experiments.runStage4(y, Z, splitIdx, cfg)
%   out = experiments.runStage4(y, Z, splitIdx, cfg, opts)
%
%   Fits four models on the in-sample window, produces genuine 1-step-ahead
%   OOS forecasts via the fixed-parameter rolling scheme (GARCH via
%   garch.rollingForecast; SV/RECH via inference.forecast.rollingPredictive),
%   scores them on PPS/QS/MSE/MAE/R2LOG/QLIKE with squared returns as the
%   volatility proxy, and compares via Diebold-Mariano (vs the proposed model)
%   and the Model Confidence Set on the per-step QLIKE loss.
%
%   Inputs
%   ------
%     y        : T-by-1 full return series (train + test), mean already removed.
%     Z        : T-by-K covariate matrix (standardised; T-by-0 if none).
%     splitIdx : last in-sample index.
%     cfg      : struct with fields .model (.nCovariates,.leverage), .smc
%                (.N,.M,.nSweeps,.proposalScale,.essThreshold,.targetEss,
%                .verbose), .forecast.J, .eval (.alphaQS,.mcsB),
%                .particleFilter.clipLogWeight, .baseSeed.
%     opts     : struct, optional — .writeOutputs (false), .outDir
%                ('results/stage4').
%
%   Output (struct `out`)
%   ---------------------
%     .modelNames     1-by-4 cellstr
%     .scores         1-by-4 struct array (pps, qs1, qs5, mse, mae, r2log, qlike)
%     .dmVsProposed   1-by-3 struct array (DM of each baseline vs SVLTRECH)
%     .mcs            modelConfidenceSet output over the 4 models
%     .forecasts      struct of per-model test-window varForecast columns
%     .testIdx        test-window indices

    arguments
        y        (:,1) double
        Z        (:,:) double
        splitIdx (1,1) double {mustBeInteger, mustBePositive}
        cfg      (1,1) struct
        opts           struct = struct()
    end

    defaults.writeOutputs = false;
    defaults.outDir       = fullfile('results', 'stage4');
    opts = mergeStruct(defaults, opts);

    utils.reproducibility(cfg.baseSeed);

    T       = numel(y);
    testIdx = (splitIdx + 1 : T)';
    yTest   = y(testIdx);
    rvProxy = max(yTest .^ 2, 1e-8);    % squared-return variance proxy (floored)
    rvSqrt  = sqrt(rvProxy);
    alphaQS = cfg.eval.alphaQS;

    K = size(Z, 2);

    smcOpts = struct( ...
        'N',             cfg.smc.N, ...
        'M',             cfg.smc.M, ...
        'nSweeps',       cfg.smc.nSweeps, ...
        'proposalScale', cfg.smc.proposalScale, ...
        'essThreshold',  cfg.smc.essThreshold, ...
        'targetEss',     cfg.smc.targetEss, ...
        'verbose',       cfg.smc.verbose, ...
        'pfOpts',        struct('clipLogWeight', cfg.particleFilter.clipLogWeight));
    pfOpts = struct('clipLogWeight', cfg.particleFilter.clipLogWeight);
    J      = cfg.forecast.J;

    modelNames = {'GARCH-t', 'GJR-t', 'SVLT', 'SVLTRECH'};
    nModels    = numel(modelNames);
    scores     = repmat(emptyScore(), 1, nModels);
    qlAll      = zeros(numel(testIdx), nModels);
    fc         = struct();

    %% --- 1. GARCH(1,1)-t ---
    fitG = garch.fitGarch(y(1:splitIdx), struct('type', 'garch', 'dist', 't'));
    rfG  = garch.rollingForecast(fitG, y, splitIdx);
    [scores(1), qlAll(:,1)] = scoreModel(rfG.sigma2Forecast, rfG.logPredDensity, ...
        fitG.nu, yTest, rvProxy, rvSqrt, alphaQS);
    fc.garch = rfG.sigma2Forecast;

    %% --- 2. GJR-GARCH(1,1)-t ---
    fitGJR = garch.fitGarch(y(1:splitIdx), struct('type', 'gjr', 'dist', 't'));
    rfGJR  = garch.rollingForecast(fitGJR, y, splitIdx);
    [scores(2), qlAll(:,2)] = scoreModel(rfGJR.sigma2Forecast, rfGJR.logPredDensity, ...
        fitGJR.nu, yTest, rvProxy, rvSqrt, alphaQS);
    fc.gjr = rfGJR.sigma2Forecast;

    %% --- 3. SVLT (SV baseline) ---
    mdlSVLT = models.SVLT();                        % default cholesky leverage
    resSVLT = inference.smc.likelihoodAnneal(mdlSVLT, y(1:splitIdx), smcOpts);
    nuSVLT  = mean(resSVLT.theta(:, 5));            % nu is index 5
    fpSVLT  = inference.forecast.rollingPredictive(mdlSVLT, y, resSVLT.theta, ...
        splitIdx, struct('J', J, 'M', cfg.smc.M, 'pfOpts', pfOpts));
    [scores(3), qlAll(:,3)] = scoreModel(fpSVLT.varForecast, fpSVLT.logPredDensity, ...
        nuSVLT, yTest, rvProxy, rvSqrt, alphaQS);
    fc.svlt = fpSVLT.varForecast;

    %% --- 4. SVLTRECH (proposed, SRN) ---
    mdlRECH = models.SVLTRECH('nCovariates', K, 'leverage', cfg.model.leverage);
    mdlRECH.setCovariates(Z(1:splitIdx, :));        % in-sample covariates for fitting
    resRECH = inference.smc.likelihoodAnneal(mdlRECH, y(1:splitIdx), smcOpts);
    nuRECH  = mean(resRECH.theta(:, 5));
    mdlRECH.setCovariates(Z);                       % FULL covariates for rolling OOS
    fpRECH  = inference.forecast.rollingPredictive(mdlRECH, y, resRECH.theta, ...
        splitIdx, struct('J', J, 'M', cfg.smc.M, 'pfOpts', pfOpts));
    [scores(4), qlAll(:,4)] = scoreModel(fpRECH.varForecast, fpRECH.logPredDensity, ...
        nuRECH, yTest, rvProxy, rvSqrt, alphaQS);
    fc.svltrech = fpRECH.varForecast;

    %% --- Diebold-Mariano: each baseline vs the proposed model (col 4) ---
    dmVsProposed = repmat(eval.dieboldMariano(qlAll(:,1), qlAll(:,4), 1), 1, nModels - 1);
    for m = 2:(nModels - 1)
        dmVsProposed(m) = eval.dieboldMariano(qlAll(:, m), qlAll(:, nModels), 1);
    end

    %% --- Model Confidence Set on per-step QLIKE loss ---
    mcs = eval.modelConfidenceSet(qlAll, struct('B', cfg.eval.mcsB));

    out.modelNames   = modelNames;
    out.scores       = scores;
    out.dmVsProposed = dmVsProposed;
    out.mcs          = mcs;
    out.forecasts    = fc;
    out.testIdx      = testIdx;

    if opts.writeOutputs
        writeStage4Outputs(out, opts.outDir);
    end
end


function sc = emptyScore()
% Fixed field order so repmat'd struct array is homogeneous.
    sc = struct('pps', NaN, 'qs1', NaN, 'qs5', NaN, ...
                'mse', NaN, 'mae', NaN, 'r2log', NaN, 'qlike', NaN);
end


function [sc, qlSeries] = scoreModel(varF, lpd, nu, yTest, rvProxy, rvSqrt, alphaQS)
% scoreModel  Five predictive scores + per-obs QLIKE for one model.
%   varF : test-window one-step variance forecasts (positive).
%   lpd  : test-window log predictive densities.
%   nu   : Student-t dof (for the quantile forecast).
    vHat   = sqrt(varF);
    tScale = sqrt((nu - 2) / nu);                  % unit-variance t scaling
    sc.pps   = eval.pps(lpd);
    sc.qs1   = eval.quantileScore(yTest, vHat .* tinv(alphaQS(1), nu) .* tScale, alphaQS(1));
    sc.qs5   = eval.quantileScore(yTest, vHat .* tinv(alphaQS(2), nu) .* tScale, alphaQS(2));
    mm       = eval.mseMae(rvSqrt, vHat);
    sc.mse   = mm.mse;
    sc.mae   = mm.mae;
    sc.r2log = eval.r2log(rvProxy, vHat);
    sc.qlike = eval.qlike(varF, rvProxy);
    qlSeries = log(varF) + rvProxy ./ varF;        % per-obs QLIKE for DM / MCS
end


function writeStage4Outputs(out, outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    rows = cell(numel(out.scores), 8);
    for m = 1:numel(out.scores)
        s = out.scores(m);
        rows(m, :) = {out.modelNames{m}, s.pps, s.qs1, s.qs5, s.mse, s.mae, s.r2log, s.qlike};
    end
    tbl = cell2table(rows, 'VariableNames', ...
        {'model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'});
    writetable(tbl, fullfile(outDir, 'scores.csv'));

    fcTbl = table(out.testIdx, out.forecasts.garch, out.forecasts.gjr, ...
                  out.forecasts.svlt, out.forecasts.svltrech, ...
        'VariableNames', {'testIdx','GARCH','GJR','SVLT','SVLTRECH'});
    writetable(fcTbl, fullfile(outDir, 'forecasts.csv'));
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tStage4Smoke.m')"`
Expected: PASS (1/1). (Allow ~1-3 min: it runs two small SMC fits.)

If `models.SVLT()` errors on construction, open `+models/SVLT.m`, read its constructor `arguments` block, and adjust the call (e.g. `models.SVLT('leverage','cholesky')`) — do not guess; match the actual signature.

- [ ] **Step 5: Checkpoint — full suite stays green**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); r = runtests('tests'); disp(table(r))"`
Expected: all tests pass (prior 65 + GARCH baselines + the 3 new files). Investigate any failure before continuing.

---

### Task 5: Data loader + network entry point

**Files:**
- Create: `+data/loadStage4.m`
- Create: `scripts/run_stage4_vietnam.m`

- [ ] **Step 1: Create `+data/loadStage4.m`**

```matlab
function ds = loadStage4(cfg)
% loadStage4  Fetch + align VN-Index and four covariates for Stage 4.
%
%   ds = data.loadStage4(cfg)
%
%   cfg.data fields: .start, .end, .trainEnd, .covariates (cellstr subset of
%   {'oil','usdvnd','btc','gold'}). Requires network + the project venv
%   (run `source setup.sh`, and pyenv pointed at .venv).
%
%   Returns struct `ds`:
%     .dates    n-by-1 datetime (trading days with a valid VN-Index close)
%     .y        n-by-1 returns, 100*log(close_t/close_{t-1})  (NaN-dropped)
%     .Z        n-by-K covariate matrix, RAW (NOT yet standardised)
%     .covNames 1-by-K cellstr
%     .splitIdx scalar = cfg.data.trainEnd (clamped to < n)
%
%   Standardisation is intentionally deferred to the caller so it can apply
%   train-only statistics after the chronological split (no leakage).

    arguments
        cfg (1,1) struct
    end

    optsP = struct('start', cfg.data.start, 'end', cfg.data.end);

    %% VN-Index close -> 100*log returns
    vnTbl = data.fetch.fetchVN('VNINDEX', optsP);
    px    = vnTbl.close;
    ret   = [NaN; 100 * log(px(2:end) ./ px(1:end-1))];
    base  = table(vnTbl.date, ret, 'VariableNames', {'date','y'});

    %% Covariates -> daily levels, joined on date
    src = struct( ...
        'oil',    'DCOILWTICO', ...        % WTI crude, USD/bbl (FRED)
        'gold',   'GOLDAMGBD228NLBM', ...  % London gold fixing (FRED)
        'usdvnd', 'DEXVNUS');              % USD/VND (FRED; verify availability)
    covNames = cfg.data.covariates;
    for c = 1:numel(covNames)
        name = covNames{c};
        switch name
            case {'oil','gold','usdvnd'}
                ft  = data.fetch.fetchFRED(src.(name), optsP);
                col = table(ft.date, ft.value, 'VariableNames', {'date', name});
            case 'btc'
                yt  = data.fetch.fetchYahoo('BTC-USD', optsP);
                col = table(yt.date, yt.close, 'VariableNames', {'date', name});
            otherwise
                error('loadStage4:unknownCovariate', 'Unknown covariate %s', name);
        end
        base = outerjoin(base, col, 'Keys', 'date', 'MergeKeys', true, 'Type', 'left');
    end

    %% Forward-fill covariates across non-trading gaps, then drop missing target
    base = sortrows(base, 'date');
    for c = 1:numel(covNames)
        base.(covNames{c}) = fillmissing(base.(covNames{c}), 'previous');
    end
    base = base(~isnan(base.y), :);
    Zraw = base{:, covNames};
    good = all(~isnan(Zraw), 2);          % drop leading rows with no prior to fill
    base = base(good, :);

    n           = height(base);
    ds.dates    = base.date;
    ds.y        = base.y;
    ds.Z        = base{:, covNames};
    ds.covNames = covNames;
    ds.splitIdx = min(cfg.data.trainEnd, n - 1);
end
```

- [ ] **Step 2: Create `scripts/run_stage4_vietnam.m`**

```matlab
%% run_stage4_vietnam  Stage 4 entry point (PROPOSED_METHODOLOGY §11.4).
%  Run after `source setup.sh`:
%    matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); run('scripts/run_stage4_vietnam.m')"

cfg = utils.loadConfig('config/smc_defaults.yaml', 'config/experiments/stage4.yaml');

%% Fetch + align
ds = data.loadStage4(cfg);
fprintf('[stage4] loaded %d obs, %d covariates (%s)\n', ...
        numel(ds.y), numel(ds.covNames), strjoin(ds.covNames, ', '));

%% Fetch-sanity anchor (NEU thesis Table 4.14: kurtosis ~7.73, std ~0.498)
fprintf('[stage4] VN-Index returns: mean=%.4f std=%.4f skew=%.4f kurt=%.4f n=%d\n', ...
        mean(ds.y), std(ds.y), skewness(ds.y), kurtosis(ds.y), numel(ds.y));

%% Leakage-safe covariate standardisation using TRAIN stats only
ZTrain = ds.Z(1:ds.splitIdx, :);
ZTest  = ds.Z(ds.splitIdx+1:end, :);
[ZTrainStd, ZTestStd] = data.preprocess.standardize(ZTrain, ZTest);
Zstd = [ZTrainStd; ZTestStd];

%% Run the pipeline
out = experiments.runStage4(ds.y, Zstd, ds.splitIdx, cfg, ...
        struct('writeOutputs', true, 'outDir', fullfile('results','stage4')));

%% Report
fprintf('\n[stage4] scores written to results/stage4/scores.csv\n');
for m = 1:numel(out.modelNames)
    s = out.scores(m);
    fprintf('  %-9s  PPS=%.3f QLIKE=%.3f MSE=%.3f MAE=%.3f R2LOG=%.3f  inMCS=%d\n', ...
        out.modelNames{m}, s.pps, s.qlike, s.mse, s.mae, s.r2log, out.mcs.inSet(m));
end
```

- [ ] **Step 3: Network/health probe (manual)**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); t = data.fetch.fetchFRED('DCOILWTICO', struct('start','2024-01-01','end','2024-02-01')); disp(height(t))"`
Expected: a small positive row count if network + FRED reachable. If it errors (offline / endpoint change), STOP the live run — the slice's correctness is already proven by `tStage4Smoke`; record the network gap and resume the real run when connectivity is available. Do NOT fabricate data.

- [ ] **Step 4: Checkpoint** — files created; live execution happens in Task 6.

---

### Task 6: Execute the slice + verify acceptance criteria

**Files:** none (execution + verification only)

- [ ] **Step 1: Full test suite green**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); r = runtests('tests'); assert(all([r.Passed])); disp('ALL GREEN')"`
Expected: `ALL GREEN`.

- [ ] **Step 2: Real Stage-4 run (network permitting)**

Run: `source setup.sh && /home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); run('scripts/run_stage4_vietnam.m')" 2>&1 | tee results/stage4/run.log`
Expected: prints the descriptive line + a score row per model; writes `results/stage4/{scores.csv,forecasts.csv}`.

- [ ] **Step 3: Verify the fetch anchor (PHASE_5_PLAN §6)**

Inspect the printed VN-Index descriptive line. Expected (NEU thesis Table 4.14, ~2,262 obs): mean ≈ 0.0149, std ≈ 0.498, skew ≈ −0.987, **kurtosis ≈ 7.73**, n ≈ 2,262. A material mismatch means the fetch is wrong (ticker / calendar / scaling) — fix `data.loadStage4` before trusting any score. If offline, this step waits.

- [ ] **Step 4: Verify slice acceptance (PHASE_5_PLAN §8)**

Confirm: (1) new tests + full suite pass; (2) `results/stage4/scores.csv` has one row per model with all finite scores; (3) the fetch anchor matches (or offline is documented); (4) a comparison is reported. Record results — including whether SVLTRECH actually beats GARCH-t — **as found**, not tuned (PROPOSED_METHODOLOGY verification note).

- [ ] **Step 5: Write the completion record**

Create `docs/PHASE_5_STAGE4_COMPLETE.md` summarising: files added, test tally, the VN-Index descriptive line vs the anchor, the score table, the DM/MCS verdict, and any deviations (e.g. offline fetch, USD/VND source fallback). Mirror the style of `docs/PHASE_3_SCAFFOLD_COMPLETE.md`.

---

## Self-review

**Spec coverage** (against `docs/PHASE_5_PLAN.md`):
- §2.1 models (GARCH-t, GJR-t, SVLT, SVLTRECH) → Task 4 ✓
- §2.1 scores (PPS/QS/MSE/MAE/R²LOG/QLIKE) + DM + MCS → Task 4 `scoreModel` + DM/MCS ✓
- §2.1 squared-return proxy → Task 4 `rvProxy` ✓
- §3 fixed-parameter rolling predictive (thin J, run PF, log-BMA) → Task 2 ✓
- §4.1 bootstrap extension → Task 1 ✓; rollingPredictive → Task 2 ✓; runStage4 → Task 4 ✓; stage4.yaml → Task 3 ✓; loadStage4 → Task 5 ✓
- §4.2 tests (`tRollingPredictive`, `tStage4Smoke`) → Tasks 2, 4 ✓ (plus `tBootstrapIncrements` for the PF extension)
- §6 fetch anchor → Task 5 print + Task 6 Step 3 ✓
- §7 network risk → Task 5 Step 3 probe + Task 6 Step 2 offline handling ✓
- §8 acceptance → Task 6 ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type/signature consistency:** `scoreModel` returns `[sc, qlSeries]` used consistently; `emptyScore()` fixes the struct-array field order so `repmat` + indexed assignment is homogeneous; `rollingPredictive` output fields (`.varForecast`, `.logPredDensity`, `.testIdx`, `.logPredDensityAll`, `.logPredPerParticle`) match the test and `runStage4`; `bootstrap` 4-output form `[logLik,hFiltered,logPredDensity,varForecast]` matches the Task-2 caller `[~,~,lpd,vf]`; `nu` index 5 valid for SVLT and SVLTRECH; all `eval.*` calls match the read signatures (`pps(lpd)`, `quantileScore(y,qHat,alpha)`, `mseMae(rvSqrt,vHat)→.mse/.mae`, `r2log(rv,vHat)`, `qlike(s2Hat,s2Act)`, `dieboldMariano(l1,l2,h)`, `modelConfidenceSet(losses,opts)`). ✓

**Known residual risks (flagged, not placeholders):** `models.SVLT()` constructor arg assumed default-cholesky (Task 4 Step 4 has a verify-and-adjust instruction); FRED `DEXVNUS` availability for USD/VND (PHASE_5_PLAN §7 fallback).
