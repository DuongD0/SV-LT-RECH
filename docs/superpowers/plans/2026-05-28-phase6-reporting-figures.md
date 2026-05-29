# Phase 6 — Reporting & Figures Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the paper-deliverable layer — tables T1–T7 and figures F1–F4 (PROPOSED_METHODOLOGY.md §12) — driven by a complete results bundle persisted by an expanded 6-model Stage-4 pipeline, emitted as CSV + LaTeX + PNG/PDF, and proven end-to-end offline on a synthetic bundle.

**Architecture:** Expand `experiments.runStage4` (via a config-driven model registry) to fit 6 models and persist a self-describing `bundle.mat` (posteriors, log-ML, standardized residuals, ω-paths, forecasts, scores, DM, MCS, descriptives). A new `+report` package consumes the saved bundle only: seven table builders + a booktabs `latexTable` renderer + four figure builders, orchestrated by `scripts/make_paper_tables.m`. No inference re-runs in the report layer.

**Tech Stack:** MATLAB R2026a (Econometrics + Statistics toolboxes), `matlab.unittest`. Reference: `PROPOSED_METHODOLOGY.md` §10/§12/§13; design spec `docs/superpowers/specs/2026-05-28-phase6-reporting-figures-design.md`.

---

## Conventions for this plan

- **MATLAB binary** is not on `PATH`: invoke `/home/d0/MATLAB/R2026a/bin/matlab`.
- **Run the whole suite:**
  `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests')"`
- **Run one test file:**
  `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tNAME.m')"`
  (Do **not** use `runtests({'tests/tNAME'})` without `.m`. Memory: `matlab-env`.)
- **First launch after reboot can hang** on MathWorks service init; if a `-batch` run produces zero output for minutes, kill the process tree and relaunch. Probe with `-batch "disp(1+1)"`.
- **Checkpoints run the relevant tests** (matching the Phase-5 plan convention). Git commits are optional; if you want version control, the repo is on `main` — branch first.
- All new MATLAB source must be `checkcode`-clean.
- Figures use `figure('Visible','off')` + `exportgraphics`; never require a display.

---

## File structure (locked before tasks)

| File | Responsibility |
|---|---|
| `+data/+preprocess/bdsTest.m` (create) | Compact correlation-integral BDS statistic + p-value |
| `+experiments/stage4Models.m` (create) | Model registry: the 6 Stage-4 models as a spec struct array |
| `+models/SVLTRECH.m` (modify) | Add `omegaPath(theta, hPath, yPath)` method for F4 |
| `+experiments/runStage4.m` (rewrite) | Registry loop → fit 6 models → assemble + persist `bundle.mat`; `out` superset-compatible |
| `+report/assertBundle.m` (create) | Validate required bundle fields with a clear error |
| `+report/latexTable.m` (create) | MATLAB `table` → booktabs LaTeX (bold-best, MCSE parens) |
| `+report/withDefaults.m` (create) | Shared figure-option defaults |
| `+report/saveFigure.m` (create) | Write a figure to `<stem>.png` + `<stem>.pdf` |
| `+report/tableT1descriptives.m` (create) | T1: descriptives + ADF/PP/KPSS + JB/ARCH-LM/BDS |
| `+report/tableT2posteriors.m` (create) | T2: posterior mean(std) per parameter per model |
| `+report/tableT3logml.m` (create) | T3: log marginal likelihoods |
| `+report/tableT4scores.m` (create) | T4: PPS/QS1/QS5/MSE/MAE/R2LOG/QLIKE per model |
| `+report/tableT5dm.m` (create) | T5: Diebold–Mariano stat + p-value vs proposed |
| `+report/tableT6mcs.m` (create) | T6: MCS membership + p-values |
| `+report/tableT7residuals.m` (create) | T7: std-resid moments + LB-Q² per model |
| `+report/figForecastBands.m` (create) | F1: 95% one-step bands over OOS returns |
| `+report/figQQ.m` (create) | F2: standardized-residual QQ vs N(0,1) |
| `+report/figCovariatePosterior.m` (create) | F3: v_z posterior densities |
| `+report/figOmegaState.m` (create) | F4: ω_t recurrent-state path |
| `scripts/make_paper_tables.m` (create) | Orchestrator: bundle → all tables + figures |
| `scripts/run_stage4_demo.m` (create) | Offline synthetic end-to-end demo |
| `tests/tBdsTest.m` (create) | BDS iid-vs-dependent sanity |
| `tests/tStage4Models.m` (create) | Registry shape/contents |
| `tests/tOmegaPath.m` (create) | ω-path shape + reproduces simulate's ω |
| `tests/tReportTables.m` (create) | T1–T7 builders + latexTable + orchestrator |
| `tests/tReportFigures.m` (create) | F1–F4 emit non-empty PNG+PDF |
| `tests/tStage4Smoke.m` (modify) | Expand to 6 models + bundle assertions |
| `tests/fixtures/syntheticBundle.m` (create) | Builds a small in-memory bundle for report tests |

---

### Task 1: Compact BDS test for nonlinearity (T1)

**Files:**
- Create: `+data/+preprocess/bdsTest.m`
- Test: `tests/tBdsTest.m`

- [ ] **Step 1: Write the failing test**

Create `tests/tBdsTest.m`:

```matlab
classdef tBdsTest < matlab.unittest.TestCase
% tBdsTest  BDS statistic: does not over-reject iid, rejects dependence.

    methods (Test)

        function iidNotRejected(testCase)
            rng(101, 'threefry');
            x = randn(1500, 1);                 % iid Gaussian -> H0 true
            out = data.preprocess.bdsTest(x, 2, 0.5);
            testCase.verifyTrue(isfinite(out.stat));
            testCase.verifyGreaterThanOrEqual(out.pValue, 0);
            testCase.verifyLessThanOrEqual(out.pValue, 1);
            testCase.verifyLessThan(abs(out.stat), 3.0);   % ~no rejection
        end

        function archDependenceRejected(testCase)
            rng(202, 'threefry');
            T = 1500; x = zeros(T, 1); h = 1;
            for t = 2:T
                h = 0.2 + 0.7 * x(t-1)^2 + 0.1 * h;     % volatility clustering
                x(t) = sqrt(h) * randn();
            end
            out = data.preprocess.bdsTest(x, 2, 0.5);
            testCase.verifyGreaterThan(abs(out.stat), 5.0);  % strong rejection
            testCase.verifyLessThan(out.pValue, 1e-3);
        end

        function embeddingDimDefaults(testCase)
            rng(7, 'threefry');
            out = data.preprocess.bdsTest(randn(400, 1));    % defaults m=2, eps=0.5
            testCase.verifyTrue(isfinite(out.stat));
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tBdsTest.m')"`
Expected: FAIL — `data.preprocess.bdsTest` does not exist.

- [ ] **Step 3: Implement `bdsTest.m`**

Create `+data/+preprocess/bdsTest.m`:

```matlab
function out = bdsTest(x, m, epsFrac)
% bdsTest  Brock-Dechert-Scheinkman test for iid (nonlinear dependence).
%
%   out = bdsTest(x)
%   out = bdsTest(x, m, epsFrac)
%
%   Tests H0: x is iid against unspecified (often nonlinear) dependence,
%   via the correlation-integral statistic of Brock, Dechert, Scheinkman &
%   LeBaron (1996). Rejection on residuals motivates the neural-network
%   correction (PROPOSED_METHODOLOGY.md §4, Table T1).
%
%   Inputs
%   ------
%     x       : T-by-1 series (e.g. returns or standardized residuals).
%     m       : embedding dimension (default 2).
%     epsFrac : neighbourhood radius as a fraction of std(x) (default 0.5).
%
%   Output (struct `out`)
%   ---------------------
%     .stat   : standardized BDS statistic (~ N(0,1) under H0).
%     .pValue : two-sided normal p-value.
%     .m, .eps: echoed settings.

    arguments
        x       (:,1) double
        m       (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(m, 2)} = 2
        epsFrac (1,1) double {mustBePositive} = 0.5
    end

    T   = numel(x);
    eps = epsFrac * std(x);

    % Pairwise indicator A(s,t) = 1 if |x_s - x_t| < eps (T-by-T, symmetric).
    A = abs(x - x.') < eps;

    % m=1 correlation integral over ordered distinct pairs.
    g  = sum(A, 2) - 1;                 % per-row neighbour count (exclude self)
    C1 = sum(g) / (T * (T - 1));

    % K: probability three points are pairwise within eps.
    K = (sum(g.^2) - sum(g)) / (T * (T - 1) * (T - 2));

    % m-dimensional correlation integral on overlapping m-histories.
    M  = T - m + 1;
    Am = true(M, M);
    for j = 0:(m - 1)
        Am = Am & A(1 + j : M + j, 1 + j : M + j);
    end
    Cm = (sum(Am(:)) - M) / (M * (M - 1));   % exclude diagonal; ordered pairs

    % BDS variance (Brock et al. 1996, eq. for dimension m).
    sigma2 = K^m + (m - 1)^2 * C1^(2 * m) - m^2 * K * C1^(2 * m - 2);
    for j = 1:(m - 1)
        sigma2 = sigma2 + 2 * (K^(m - j) * C1^(2 * j));
    end
    sigma2 = 4 * sigma2;

    out.stat   = sqrt(T) * (Cm - C1^m) / sqrt(max(sigma2, 1e-12));
    out.pValue = 2 * (1 - normcdf(abs(out.stat)));
    out.m      = m;
    out.eps    = eps;
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tBdsTest.m')"`
Expected: PASS (3/3).

- [ ] **Step 5: Checkpoint** — `checkcode +data/+preprocess/bdsTest.m` clean; Task-1 tests green.

---

### Task 2: Stage-4 model registry

**Files:**
- Create: `+experiments/stage4Models.m`
- Test: `tests/tStage4Models.m`

- [ ] **Step 1: Write the failing test**

Create `tests/tStage4Models.m`:

```matlab
classdef tStage4Models < matlab.unittest.TestCase
% tStage4Models  The 6-model Stage-4 registry has the expected shape.

    methods (Test)
        function registryShape(testCase)
            reg = experiments.stage4Models();
            names = {reg.name};
            testCase.verifyEqual(names, ...
                {'GARCH-t','GJR-t','SVLT','SVLTRECH-SRN', ...
                 'SVLTRECH-LSTM','SVLTRECH-GRU'});
            testCase.verifyEqual(sum([reg.isProposed]), 1);   % exactly one proposed
            testCase.verifyEqual(reg(strcmp(names,'SVLTRECH-SRN')).isProposed, true);
        end

        function svConstructorsBuild(testCase)
            reg = experiments.stage4Models();
            K   = 2;
            for i = 1:numel(reg)
                if strcmp(reg(i).kind, 'sv')
                    mdl = reg(i).ctor(K, 'cholesky');
                    testCase.verifyTrue(isa(mdl, 'models.Model'));
                end
            end
        end

        function garchEntriesCarryType(testCase)
            reg = experiments.stage4Models();
            g   = reg(strcmp({reg.name}, 'GJR-t'));
            testCase.verifyEqual(g.kind, 'garch');
            testCase.verifyEqual(g.garchType, 'gjr');
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tStage4Models.m')"`
Expected: FAIL — `experiments.stage4Models` does not exist.

- [ ] **Step 3: Implement `stage4Models.m`**

Create `+experiments/stage4Models.m`:

```matlab
function reg = stage4Models()
% stage4Models  Registry of the six Stage-4 comparison models.
%
%   reg = experiments.stage4Models()
%
%   Single source of truth for which models the Stage-4 pipeline fits and
%   reports. Each entry:
%     .name       display label
%     .kind       'garch' (Econometrics-Toolbox MLE) | 'sv' (SMC)
%     .garchType  'garch' | 'gjr'   (kind=='garch' only; '' otherwise)
%     .ctor       @(K, leverage) -> models.Model  (kind=='sv' only; [] else)
%     .isProposed true for the flagship vs which baselines are DM-compared
%
%   See PROPOSED_METHODOLOGY.md §7-8 and the Phase-6 design spec §4.2.

    reg        = entry('GARCH-t',       'garch', 'garch', [], false);
    reg(end+1) = entry('GJR-t',         'garch', 'gjr',   [], false);
    reg(end+1) = entry('SVLT',          'sv', '', @(~, ~)    models.SVLT(),                                      false);
    reg(end+1) = entry('SVLTRECH-SRN',  'sv', '', @(K, lev)  models.SVLTRECH('nCovariates', K, 'leverage', lev),     true);
    reg(end+1) = entry('SVLTRECH-LSTM', 'sv', '', @(K, lev)  models.SVLTLSTMRECH('nCovariates', K, 'leverage', lev), false);
    reg(end+1) = entry('SVLTRECH-GRU',  'sv', '', @(K, lev)  models.SVLTGRURECH('nCovariates', K, 'leverage', lev),  false);
end


function e = entry(name, kind, garchType, ctor, isProposed)
    e = struct('name', name, 'kind', kind, 'garchType', garchType, ...
               'ctor', ctor, 'isProposed', isProposed);
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tStage4Models.m')"`
Expected: PASS (3/3).

- [ ] **Step 5: Checkpoint** — registry tests green; `checkcode` clean.

---

### Task 3: `omegaPath` method on the SRN flagship (F4 input)

**Files:**
- Modify: `+models/SVLTRECH.m` (add one method)
- Test: `tests/tOmegaPath.m`

Interpretability (F4) is the SRN flagship's selling point (§8.2). We expose the
deterministic ω-path for that model; extending to LSTM/GRU is a trivial follow-on.

- [ ] **Step 1: Write the failing test**

Create `tests/tOmegaPath.m`:

```matlab
classdef tOmegaPath < matlab.unittest.TestCase
% tOmegaPath  SVLTRECH.omegaPath shape + first value (= beta_0).

    methods (Test)
        function shapeAndFirstValue(testCase)
            rng(5, 'threefry');
            mdl   = models.SVLTRECH('nCovariates', 0);
            theta = mdl.samplePrior(1);
            theta(3) = abs(theta(3));                 % sigma_eta > 0
            [y, h] = mdl.simulate(theta, 120);
            omega  = mdl.omegaPath(theta, h, y);
            testCase.verifySize(omega, [120, 1]);
            s = mdl.unpack(theta);
            testCase.verifyEqual(omega(1), s.beta_0, 'RelTol', 1e-12);
            testCase.verifyTrue(all(isfinite(omega)));
        end

        function withCovariates(testCase)
            rng(6, 'threefry');
            K   = 2; T = 100;
            Z   = randn(T, K);
            mdl = models.SVLTRECH('nCovariates', K, 'covariates', Z);
            theta = mdl.samplePrior(1); theta(3) = abs(theta(3));
            [y, h] = mdl.simulate(theta, T);
            omega  = mdl.omegaPath(theta, h, y);
            testCase.verifySize(omega, [T, 1]);
            testCase.verifyTrue(all(isfinite(omega)));
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tOmegaPath.m')"`
Expected: FAIL — no `omegaPath` method on `models.SVLTRECH`.

- [ ] **Step 3: Add the method to `+models/SVLTRECH.m`**

Insert this method into the `methods` block of `+models/SVLTRECH.m`, immediately
after the `simulate` method's closing `end`, before the `%% ---- Packing` section:

```matlab
        %% ---- Interpretability ------------------------------------

        function omega = omegaPath(obj, theta, hPath, yPath)
        % omega = omegaPath(theta, hPath, yPath)
        % Deterministic recurrent-state path omega_t given a parameter
        % vector and a (filtered) latent-/return-path. Replays the same SRN
        % recurrence as `simulate`. omega_1 == beta_0 (s_1 == 0). For F4.
            arguments
                obj
                theta (1,:) double
                hPath (:,1) double
                yPath (:,1) double
            end
            s = obj.unpack(theta);
            T = numel(hPath);
            omega = zeros(T, 1);
            omega(1) = s.beta_0;
            sPrev = 0; omegaPrev = s.beta_0;
            for t = 2:T
                if obj.nCovariates > 0
                    zRow = obj.Z(t - 1, :);
                else
                    zRow = [];
                end
                xRow = [hPath(t - 1), yPath(t - 1), omegaPrev, zRow];
                w    = struct('v',   [s.v_h, s.v_r, s.v_omega, s.v_z(:)'], ...
                              'w_h', s.w_h, 'b', s.b);
                sNew      = cells.srn(xRow, sPrev, w);
                omega(t)  = s.beta_0 + s.beta_1 * sNew;
                sPrev     = sNew;
                omegaPrev = omega(t);
            end
        end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tOmegaPath.m')"`
Expected: PASS (2/2).

- [ ] **Step 5: Checkpoint — regression-guard SVLTRECH**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tSvltRoundTrip.m')"`
Expected: PASS (the round-trip test still passes; the new method is additive).

---

### Task 4: Rewrite `runStage4` — registry loop, 6 models, bundle persistence

**Files:**
- Rewrite: `+experiments/runStage4.m`
- Modify: `tests/tStage4Smoke.m`

This is the core task. The pipeline now loops over `experiments.stage4Models()`,
fits each model, and assembles a complete `bundle` struct (design spec §4.1). The
public `out` keeps every existing field (now 6-wide) and adds `out.bundle`.

- [ ] **Step 1: Update the smoke test to the 6-model contract + bundle**

Replace the body of `tests/tStage4Smoke.m`'s `pipelineProducesScoresForAllModels`
test method with the version below (keep the `methods (TestMethodSetup)`
`requireEcon` block unchanged):

```matlab
        function pipelineProducesScoresForAllModels(testCase)
            rng(99, 'threefry');

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
            cfg.smc      = struct('N', 200, 'M', 30, 'nSweeps', 2, ...
                                  'proposalScale', 0.5, 'essThreshold', 0.5, ...
                                  'targetEss', 0.5, 'verbose', false);
            cfg.forecast = struct('J', 15);
            cfg.eval     = struct('alphaQS', [0.01, 0.05], 'mcsB', 200);
            cfg.particleFilter = struct('clipLogWeight', -50);
            cfg.baseSeed = 20260516;

            out = experiments.runStage4(y, Z, splitIdx, cfg);

            expected = {'GARCH-t','GJR-t','SVLT','SVLTRECH-SRN', ...
                        'SVLTRECH-LSTM','SVLTRECH-GRU'};
            testCase.verifyEqual(out.modelNames, expected);

            S = out.scores;
            testCase.verifyEqual(numel(S), 6);
            for m = 1:numel(S)
                testCase.verifyTrue(isfinite(S(m).pps));
                testCase.verifyTrue(isfinite(S(m).qlike));
                testCase.verifyTrue(isfinite(S(m).mse));
                testCase.verifyTrue(isfinite(S(m).r2log));
            end

            testCase.verifyGreaterThanOrEqual(sum(out.mcs.inSet), 1);
            testCase.verifyEqual(numel(out.dmVsProposed), 5);   % 6 - 1 vs proposed

            % Bundle contract
            b = out.bundle;
            testCase.verifyEqual(numel(b.models), 6);
            testCase.verifyTrue(isfield(b, 'descriptives'));
            testCase.verifyTrue(isfield(b.descriptives, 'bds'));
            % SV models carry posterior particles + log-ML; GARCH do not.
            srn = b.models(strcmp({b.models.name}, 'SVLTRECH-SRN'));
            testCase.verifyTrue(isfinite(srn.logMarginalLik));
            testCase.verifySize(srn.theta, [cfg.smc.N, 12 + K]);
            testCase.verifyEqual(numel(srn.stdResid), splitIdx);
            testCase.verifyEqual(numel(srn.omegaPath), splitIdx);
        end
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tStage4Smoke.m')"`
Expected: FAIL — current `runStage4` returns 4 models and no `out.bundle`.

- [ ] **Step 3: Rewrite `+experiments/runStage4.m`**

Replace the entire file with:

```matlab
function out = runStage4(y, Z, splitIdx, cfg, opts)
% runStage4  Stage-4 empirical pipeline: fit 6 models, OOS-forecast, score,
%            compare, and assemble a complete results bundle.
%
%   out = experiments.runStage4(y, Z, splitIdx, cfg)
%   out = experiments.runStage4(y, Z, splitIdx, cfg, opts)
%
%   Loops over experiments.stage4Models(): GARCH-t, GJR-t (Econometrics MLE +
%   garch.rollingForecast); SVLT, SVLTRECH-{SRN,LSTM,GRU} (likelihoodAnneal +
%   inference.forecast.rollingPredictive). Scores PPS/QS/MSE/MAE/R2LOG/QLIKE on
%   squared-return proxy; Diebold-Mariano each baseline vs the proposed model
%   (SVLTRECH-SRN) and the Model Confidence Set on per-step QLIKE. Persists a
%   self-describing bundle (design spec §4.1) consumed by the +report layer.
%
%   cfg fields: .model(.nCovariates,.leverage), .smc(.N,.M,.nSweeps,
%   .proposalScale,.essThreshold,.targetEss,.verbose), .forecast.J,
%   .eval(.alphaQS,.mcsB), .particleFilter.clipLogWeight, .baseSeed.
%   opts (optional): .writeOutputs (false), .outDir ('results/stage4'),
%                    .expName ('stage4').
%
%   Output `out`: .modelNames, .scores (struct array), .dmVsProposed,
%   .mcs, .forecasts, .testIdx, .bundle (the full persisted struct).

    arguments
        y        (:,1) double
        Z        (:,:) double
        splitIdx (1,1) double {mustBeInteger, mustBePositive}
        cfg      (1,1) struct
        opts           struct = struct()
    end

    defaults.writeOutputs = false;
    defaults.outDir       = fullfile('results', 'stage4');
    defaults.expName      = 'stage4';
    opts = mergeStruct(defaults, opts);

    utils.reproducibility(cfg.baseSeed);

    T       = numel(y);
    testIdx = (splitIdx + 1 : T)';
    yTest   = y(testIdx);
    rvProxy = max(yTest .^ 2, 1e-8);
    rvSqrt  = sqrt(rvProxy);
    alphaQS = cfg.eval.alphaQS;
    K       = size(Z, 2);

    smcOpts = struct( ...
        'N', cfg.smc.N, 'M', cfg.smc.M, 'nSweeps', cfg.smc.nSweeps, ...
        'proposalScale', cfg.smc.proposalScale, 'essThreshold', cfg.smc.essThreshold, ...
        'targetEss', cfg.smc.targetEss, 'verbose', cfg.smc.verbose, ...
        'pfOpts', struct('clipLogWeight', cfg.particleFilter.clipLogWeight));
    pfOpts = struct('clipLogWeight', cfg.particleFilter.clipLogWeight);
    J      = cfg.forecast.J;

    reg        = experiments.stage4Models();
    nModels    = numel(reg);
    modelNames = {reg.name};
    proposedIx = find([reg.isProposed], 1);

    scores  = repmat(emptyScore(), 1, nModels);
    qlAll   = zeros(numel(testIdx), nModels);
    models_ = repmat(emptyModelRecord(), 1, nModels);
    fc      = struct();

    for m = 1:nModels
        rec = fitOneModel(reg(m), y, Z, splitIdx, K, smcOpts, pfOpts, J, cfg);
        [scores(m), qlAll(:, m)] = scoreModel( ...
            rec.varForecast, rec.logPredDensity, rec.nu, yTest, rvProxy, rvSqrt, alphaQS);
        models_(m) = rec;
        fc.(matlab.lang.makeValidName(reg(m).name)) = rec.varForecast;
    end

    %% Diebold-Mariano: each non-proposed model vs the proposed model.
    others = setdiff(1:nModels, proposedIx);
    dmVsProposed = repmat(eval.dieboldMariano(qlAll(:,others(1)), qlAll(:,proposedIx), 1), ...
                          1, numel(others));
    for k = 1:numel(others)
        dmVsProposed(k) = eval.dieboldMariano(qlAll(:, others(k)), qlAll(:, proposedIx), 1);
    end
    dmBaselines = modelNames(others);

    %% Model Confidence Set on per-step QLIKE loss.
    mcs = eval.modelConfidenceSet(qlAll, struct('B', cfg.eval.mcsB));

    %% In-sample descriptives + diagnostics (T1 record).
    descr = describeSeries(y(1:splitIdx));

    %% Assemble bundle (design spec §4.1).
    bundle.meta = struct('expName', opts.expName, 'date', datestr(now, 'yyyy-mm-dd'), ...
                         'cfg', cfg, 'baseSeed', cfg.baseSeed, 'modelNames', {modelNames});
    bundle.data = struct('y', y, 'Z', Z, 'splitIdx', splitIdx, 'testIdx', testIdx, ...
                         'nCovariates', K);
    bundle.descriptives = descr;
    bundle.models       = models_;
    bundle.comparison   = struct('modelNames', {modelNames}, 'scores', scores, ...
                                 'qlAll', qlAll, 'dmVsProposed', dmVsProposed, ...
                                 'dmBaselines', {dmBaselines}, 'mcs', mcs, ...
                                 'proposedName', modelNames{proposedIx});

    out.modelNames   = modelNames;
    out.scores       = scores;
    out.dmVsProposed = dmVsProposed;
    out.mcs          = mcs;
    out.forecasts    = fc;
    out.testIdx      = testIdx;
    out.bundle       = bundle;

    if opts.writeOutputs
        writeStage4Outputs(bundle, opts.outDir);
    end
end


function rec = fitOneModel(spec, y, Z, splitIdx, K, smcOpts, pfOpts, J, cfg)
% Fit one registry entry; return a fully-populated model record.
    rec = emptyModelRecord();
    rec.name = spec.name;
    rec.type = spec.kind;

    if strcmp(spec.kind, 'garch')
        fit = garch.fitGarch(y(1:splitIdx), struct('type', spec.garchType, 'dist', 't'));
        rf  = garch.rollingForecast(fit, y, splitIdx);
        rec.nu             = fit.nu;
        rec.varForecast    = rf.sigma2Forecast;
        rec.logPredDensity = rf.logPredDensity;
        rec.logMarginalLik = NaN;                       % MLE, not SMC
        rec.sigma2InSample = fit.condVar;
        rec.stdResid       = standardizeResid(y(1:splitIdx), fit.condVar, fit.nu);
        rec.omegaPath      = [];
        [rec.paramNames, rec.posteriorMean, rec.posteriorStd] = garchParams(fit, spec.garchType);
        rec.theta          = [];
        return;
    end

    % --- SV / RECH model ---
    mdl = spec.ctor(K, cfg.model.leverage);
    if ismethod(mdl, 'setCovariates') && mdl.nCovariates > 0
        mdl.setCovariates(Z(1:splitIdx, :));
    end
    res      = inference.smc.likelihoodAnneal(mdl, y(1:splitIdx), smcOpts);
    thetaBar = mean(res.theta, 1);
    nu       = thetaBar(5);

    % In-sample filtered states -> sigma2, std residuals, omega (RECH only).
    [~, hF] = inference.pf.bootstrap(mdl, y(1:splitIdx), thetaBar, ...
        mergeStruct(struct('M', smcOpts.M, 'returnPath', true), pfOpts));
    sigma2InSample = exp(hF);

    if ismethod(mdl, 'setCovariates') && mdl.nCovariates > 0
        mdl.setCovariates(Z);                            % full series for OOS roll
    end
    fp = inference.forecast.rollingPredictive(mdl, y, res.theta, splitIdx, ...
        struct('J', J, 'M', smcOpts.M, 'pfOpts', pfOpts));

    rec.nu             = nu;
    rec.varForecast    = fp.varForecast;
    rec.logPredDensity = fp.logPredDensity;
    rec.logMarginalLik = res.logMarginalLik;
    rec.sigma2InSample = sigma2InSample;
    rec.stdResid       = standardizeResid(y(1:splitIdx), sigma2InSample, nu);
    rec.theta          = res.theta;
    rec.paramNames     = mdl.paramNames();
    rec.posteriorMean  = mean(res.theta, 1);
    rec.posteriorStd   = std(res.theta, 0, 1);
    if ismethod(mdl, 'omegaPath')
        rec.omegaPath = mdl.omegaPath(thetaBar, hF, y(1:splitIdx));
    else
        rec.omegaPath = [];
    end
end


function rec = emptyModelRecord()
    rec = struct('name', '', 'type', '', 'nu', NaN, ...
                 'varForecast', [], 'logPredDensity', [], 'logMarginalLik', NaN, ...
                 'sigma2InSample', [], 'stdResid', [], 'omegaPath', [], ...
                 'theta', [], 'paramNames', {{}}, 'posteriorMean', [], 'posteriorStd', []);
end


function sc = emptyScore()
    sc = struct('pps', NaN, 'qs1', NaN, 'qs5', NaN, ...
                'mse', NaN, 'mae', NaN, 'r2log', NaN, 'qlike', NaN);
end


function [sc, qlSeries] = scoreModel(varF, lpd, nu, yTest, rvProxy, rvSqrt, alphaQS)
    vHat   = sqrt(varF);
    tScale = sqrt((nu - 2) / nu);
    sc.pps   = eval.pps(lpd);
    sc.qs1   = eval.quantileScore(yTest, vHat .* tinv(alphaQS(1), nu) .* tScale, alphaQS(1));
    sc.qs5   = eval.quantileScore(yTest, vHat .* tinv(alphaQS(2), nu) .* tScale, alphaQS(2));
    mm       = eval.mseMae(rvSqrt, vHat);
    sc.mse   = mm.mse;
    sc.mae   = mm.mae;
    sc.r2log = eval.r2log(rvProxy, vHat);
    sc.qlike = eval.qlike(varF, rvProxy);
    qlSeries = log(varF) + rvProxy ./ varF;
end


function e = standardizeResid(y, sigma2, nu)
% Normalised residual: Phi^{-1}(F_nu(unit-variance-eps * sqrt(nu/(nu-2)))).
    epsUnit = y ./ sqrt(sigma2);
    if isinf(nu)
        e = epsUnit;
    else
        e = norminv(tcdf(epsUnit .* sqrt(nu / (nu - 2)), nu));
    end
    e(~isfinite(e)) = NaN;
end


function [names, mu, sd] = garchParams(fit, garchType)
% Point estimates as a degenerate posterior (std = NaN; the toolbox object
% does not expose standard errors here). Order: omega, alpha, beta, [gamma], nu.
    names = {'omega', 'alpha', 'beta'};
    mu    = [fit.omega, fit.alpha, fit.beta];
    if strcmp(garchType, 'gjr')
        names{end+1} = 'gamma';
        mu(end+1)    = fit.gamma;
    end
    names{end+1} = 'nu';
    mu(end+1)    = fit.nu;
    sd = nan(size(mu));
end


function descr = describeSeries(y)
    rd = data.preprocess.residualDiagnostics(y);
    st = data.preprocess.stationarityTests(y);
    bd = data.preprocess.bdsTest(y);
    descr = struct( ...
        'n', numel(y), 'mean', mean(y), 'std', std(y), ...
        'skewness', skewness(y), 'kurtosis', kurtosis(y), ...
        'min', min(y), 'max', max(y), ...
        'adf', st.adf, 'pp', st.pp, 'kpss', st.kpss, ...
        'jb', rd.jb, 'archlm', rd.archlm, 'bds', bd);
end


function writeStage4Outputs(bundle, outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    save(fullfile(outDir, 'bundle.mat'), 'bundle');

    names = bundle.comparison.modelNames;
    S     = bundle.comparison.scores;
    rows  = cell(numel(S), 8);
    for m = 1:numel(S)
        s = S(m);
        rows(m, :) = {names{m}, s.pps, s.qs1, s.qs5, s.mse, s.mae, s.r2log, s.qlike};
    end
    tbl = cell2table(rows, 'VariableNames', ...
        {'model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'});
    writetable(tbl, fullfile(outDir, 'scores.csv'));
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
Expected: PASS (1/1). Allow ~3-6 min: it runs four small SMC fits.

If `models.SVLT()` errors on the `spec.ctor(K, lev)` call, confirm the registry's
SVLT ctor ignores its args (`@(~,~) models.SVLT()`) — the working `runStage4`
already constructs `models.SVLT()` with no args, so this is correct.

- [ ] **Step 5: Checkpoint — full suite stays green**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); r = runtests('tests'); disp(table(r))"`
Expected: all pass (prior suite + new BDS/registry/omega tests). Investigate any failure before continuing.

---

### Task 5: Synthetic-bundle fixture + `report.assertBundle`

**Files:**
- Create: `tests/fixtures/syntheticBundle.m`
- Create: `+report/assertBundle.m`

The fixture lets every report test run **without** SMC — it hand-builds a bundle
with the same schema `runStage4` persists, at trivial size.

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/syntheticBundle.m`:

```matlab
function bundle = syntheticBundle()
% syntheticBundle  Small in-memory results bundle matching runStage4's schema.
%   Two SV models + two GARCH models, K=2 covariates, T=60, split=40.
%   Deterministic (fixed rng) so report tests are reproducible.

    rng(2026, 'threefry');
    T = 60; splitIdx = 40; K = 2;
    y = 0.5 * randn(T, 1);
    Z = randn(T, K);
    testIdx = (splitIdx + 1 : T)';
    nTest   = numel(testIdx);

    names = {'GARCH-t','GJR-t','SVLTRECH-SRN','SVLTRECH-LSTM'};
    nM    = numel(names);

    models_ = repmat(modelRec(), 1, nM);
    for m = 1:nM
        r = modelRec();
        r.name = names{m};
        r.nu   = 6 + m;
        r.varForecast    = 0.2 + 0.05 * rand(nTest, 1);
        r.logPredDensity = -1 - 0.1 * rand(nTest, 1);
        r.sigma2InSample = 0.2 + 0.05 * rand(splitIdx, 1);
        r.stdResid       = randn(splitIdx, 1);
        if startsWith(names{m}, 'SVLTRECH')
            r.type           = 'sv';
            r.logMarginalLik = -100 - m;
            r.theta          = [randn(80, 5), 6 + randn(80, 1), ...
                                0.1 * randn(80, 6 + K)];     % 12 + K cols
            r.paramNames     = svParamNames(K);
            r.posteriorMean  = mean(r.theta, 1);
            r.posteriorStd   = std(r.theta, 0, 1);
            r.omegaPath      = 0.05 + 0.01 * randn(splitIdx, 1);
        else
            r.type           = 'garch';
            r.logMarginalLik = NaN;
            r.paramNames     = {'omega','alpha','beta','nu'};
            r.posteriorMean  = [0.02, 0.08, 0.88, r.nu];
            r.posteriorStd   = nan(1, 4);
            r.theta          = [];
            r.omegaPath      = [];
        end
        models_(m) = r;
    end

    qlAll = 0.5 + 0.1 * rand(nTest, nM);
    scores = repmat(scoreRec(), 1, nM);
    for m = 1:nM
        scores(m) = struct('pps', 1 + 0.1*m, 'qs1', 0.03 + 0.001*m, ...
            'qs5', 0.05 + 0.001*m, 'mse', 0.1 + 0.01*m, 'mae', 0.2 + 0.01*m, ...
            'r2log', 0.7 + 0.05*m, 'qlike', -1 + 0.05*m);
    end
    dm = repmat(struct('statistic', 1.2, 'pValue', 0.2, 'meanDiff', 0.01, ...
                       'bandwidth', 3), 1, nM - 1);
    mcs = struct('inSet', [false, false, true, true], ...
                 'pValues', [0.02, 0.04, 0.9, 0.6], ...
                 'eliminationOrder', [1 2 4 3], 'statistic', 'Tmax');

    bundle.meta = struct('expName', 'synthetic', 'date', '2026-05-28', ...
                         'baseSeed', 2026, 'modelNames', {names});
    bundle.data = struct('y', y, 'Z', Z, 'splitIdx', splitIdx, ...
                         'testIdx', testIdx, 'nCovariates', K);
    bundle.descriptives = struct('n', splitIdx, 'mean', mean(y(1:splitIdx)), ...
        'std', std(y(1:splitIdx)), 'skewness', skewness(y(1:splitIdx)), ...
        'kurtosis', kurtosis(y(1:splitIdx)), 'min', min(y), 'max', max(y), ...
        'adf', pv(0.01), 'pp', pv(0.01), 'kpss', pv(0.1), ...
        'jb', pv(0.001), 'archlm', pv(0.001), ...
        'bds', struct('stat', 6.2, 'pValue', 1e-9, 'm', 2, 'eps', 0.25));
    bundle.models     = models_;
    bundle.comparison = struct('modelNames', {names}, 'scores', scores, ...
        'qlAll', qlAll, 'dmVsProposed', dm, 'dmBaselines', {names(1:end-1)}, ...
        'mcs', mcs, 'proposedName', 'SVLTRECH-SRN');
end

function r = modelRec()
    r = struct('name','','type','','nu',NaN,'varForecast',[],'logPredDensity',[], ...
        'logMarginalLik',NaN,'sigma2InSample',[],'stdResid',[],'omegaPath',[], ...
        'theta',[],'paramNames',{{}},'posteriorMean',[],'posteriorStd',[]);
end
function s = scoreRec()
    s = struct('pps',NaN,'qs1',NaN,'qs5',NaN,'mse',NaN,'mae',NaN,'r2log',NaN,'qlike',NaN);
end
function p = pv(val); p = struct('h', val < 0.05, 'pValue', val); end
function nm = svParamNames(K)
    base = {'mu','phi','sigma_eta','rho','nu','beta_0','beta_1','v_h','v_r','v_omega'};
    cov  = arrayfun(@(k) sprintf('v_z_%d', k), 1:K, 'UniformOutput', false);
    nm   = [base, cov, {'w_h','b'}];
end
```

- [ ] **Step 2: Create `+report/assertBundle.m`**

```matlab
function assertBundle(bundle)
% assertBundle  Validate the top-level shape of a results bundle.
%   Throws a clear error if a required field is missing. Called once by
%   scripts/make_paper_tables before any builder runs.
    arguments
        bundle (1,1) struct
    end
    req = {'meta','data','descriptives','models','comparison'};
    for k = 1:numel(req)
        assert(isfield(bundle, req{k}), 'report:assertBundle:missingField', ...
            'bundle is missing required field "%s"', req{k});
    end
    assert(~isempty(bundle.models), 'report:assertBundle:noModels', ...
        'bundle.models is empty');
    cmpReq = {'modelNames','scores','qlAll','dmVsProposed','mcs','proposedName'};
    for k = 1:numel(cmpReq)
        assert(isfield(bundle.comparison, cmpReq{k}), ...
            'report:assertBundle:missingComparisonField', ...
            'bundle.comparison is missing "%s"', cmpReq{k});
    end
end
```

- [ ] **Step 3: Smoke-check the fixture loads + passes assertBundle**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); addpath('tests/fixtures'); b = syntheticBundle(); report.assertBundle(b); disp(numel(b.models))"`
Expected: prints `4`, no error.

- [ ] **Step 4: Checkpoint** — fixture + assertBundle load cleanly; `checkcode` clean.

---

### Task 6: `report.latexTable` — booktabs renderer

**Files:**
- Create: `+report/latexTable.m`
- Test: `tests/tReportTables.m` (first two tests; the rest are added in Tasks 7–10, 13)

- [ ] **Step 1: Write the failing test**

Create `tests/tReportTables.m`:

```matlab
classdef tReportTables < matlab.unittest.TestCase
% tReportTables  Report table builders + LaTeX renderer over a synthetic bundle.

    properties
        bundle
    end

    methods (TestClassSetup)
        function loadFixture(testCase)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, 'fixtures'));
            testCase.bundle = syntheticBundle();
        end
    end

    methods (Test)

        function latexBoldsBestPerColumn(testCase)
            T = table({'A';'B';'C'}, [1.0;0.5;0.8], [2.0;3.0;1.0], ...
                'VariableNames', {'model','QLIKE','LogML'});
            tex = report.latexTable(T, struct( ...
                'lowerBetter', {{'QLIKE'}}, 'higherBetter', {{'LogML'}}, ...
                'caption', 'Demo', 'label', 'tab:demo'));
            testCase.verifyClass(tex, 'char');
            testCase.verifySubstring(tex, '\begin{tabular}');
            testCase.verifySubstring(tex, '\textbf{0.5');   % min QLIKE bolded
            testCase.verifySubstring(tex, '\textbf{3');      % max LogML bolded
            testCase.verifySubstring(tex, '\bottomrule');
        end

        function latexRendersNaNAsDash(testCase)
            T = table({'A'}, NaN, 'VariableNames', {'model','QLIKE'});
            tex = report.latexTable(T, struct('lowerBetter', {{'QLIKE'}}));
            testCase.verifySubstring(tex, '--');
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: FAIL — `report.latexTable` does not exist.

- [ ] **Step 3: Implement `+report/latexTable.m`**

```matlab
function tex = latexTable(T, opts)
% latexTable  Render a MATLAB table as a booktabs LaTeX tabular.
%
%   tex = report.latexTable(T, opts)
%
%   opts (all optional):
%     .lowerBetter   cellstr of numeric columns where the min is bolded
%     .higherBetter  cellstr of numeric columns where the max is bolded
%     .caption       table caption
%     .label         \label{...}
%     .precision     sig digits for numeric cells (default 3)
%     .file          if set, also writes the .tex to this path
%
%   NaN / empty numeric cells render as "--". Returns the LaTeX char.

    arguments
        T    table
        opts struct = struct()
    end
    defaults.lowerBetter  = {};
    defaults.higherBetter = {};
    defaults.caption      = '';
    defaults.label        = '';
    defaults.precision    = 3;
    defaults.file         = '';
    opts = mergeStruct(defaults, opts);

    vars = T.Properties.VariableNames;
    nCol = numel(vars);
    nRow = height(T);
    fmt  = sprintf('%%.%dg', opts.precision);

    bestRow = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for c = 1:nCol
        v = vars{c};
        if iscell(T.(v)) || ~isnumeric(T.(v)); continue; end
        col = T.(v);
        if ismember(v, opts.lowerBetter)
            [~, bestRow(v)] = min(col);
        elseif ismember(v, opts.higherBetter)
            [~, bestRow(v)] = max(col);
        end
    end

    L = strings(0, 1);
    L(end+1) = "\begin{table}[htbp]";
    L(end+1) = "\centering";
    if ~isempty(opts.caption); L(end+1) = "\caption{" + string(opts.caption) + "}"; end
    if ~isempty(opts.label);   L(end+1) = "\label{"  + string(opts.label)   + "}"; end
    L(end+1) = "\begin{tabular}{" + string(repmat('l', 1, nCol)) + "}";
    L(end+1) = "\toprule";
    L(end+1) = strjoin(escapeCells(vars), " & ") + " \\";
    L(end+1) = "\midrule";
    for r = 1:nRow
        cells = strings(1, nCol);
        for c = 1:nCol
            v = vars{c};
            cells(c) = fmtCell(T.(v), r, v, fmt, bestRow);
        end
        L(end+1) = strjoin(cells, " & ") + " \\"; %#ok<AGROW>
    end
    L(end+1) = "\bottomrule";
    L(end+1) = "\end{tabular}";
    L(end+1) = "\end{table}";

    tex = char(strjoin(L, newline));
    if ~isempty(opts.file)
        fid = fopen(opts.file, 'w'); fwrite(fid, tex); fclose(fid);
    end
end


function s = fmtCell(col, r, v, fmt, bestRow)
    if iscell(col)
        c = escapeCells(col(r));
        s = c(1);
        return;
    end
    if ~isnumeric(col)
        s = string(col(r)); return;
    end
    x = col(r);
    if isempty(x) || ~isfinite(x)
        s = "--"; return;
    end
    s = string(sprintf(fmt, x));
    if isKey(bestRow, v) && bestRow(v) == r
        s = "\textbf{" + s + "}";
    end
end


function out = escapeCells(c)
    out = strings(1, numel(c));
    for k = 1:numel(c)
        out(k) = replace(string(c{k}), "_", "\_");
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

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: PASS (2/2).

- [ ] **Step 5: Checkpoint** — latexTable tests green; `checkcode` clean.

---

### Task 7: T1 descriptives table

**Files:**
- Create: `+report/tableT1descriptives.m`
- Test: add to `tests/tReportTables.m`

- [ ] **Step 1: Add the failing test**

Append this method to `tests/tReportTables.m`'s `methods (Test)` block:

```matlab
        function t1HasDescriptivesAndDiagnostics(testCase)
            T = report.tableT1descriptives(testCase.bundle);
            testCase.verifyClass(T, 'table');
            stats = T.Statistic;
            testCase.verifyTrue(any(strcmp(stats, 'Kurtosis')));
            testCase.verifyTrue(any(strcmp(stats, 'ADF p-value')));
            testCase.verifyTrue(any(strcmp(stats, 'BDS stat')));
            testCase.verifyTrue(all(isfinite(T.Value)));
        end
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: FAIL — `report.tableT1descriptives` does not exist.

- [ ] **Step 3: Implement `+report/tableT1descriptives.m`**

```matlab
function T = tableT1descriptives(bundle)
% tableT1descriptives  T1: descriptive stats + stationarity + nonlinearity.
%
%   T = report.tableT1descriptives(bundle)
%
%   One row per statistic, a Value column (single-series Stage 4). Mirrors
%   PROPOSED_METHODOLOGY.md §12 Table T1: moments + ADF/PP/KPSS + JB +
%   ARCH-LM + BDS. Reads bundle.descriptives only.

    arguments
        bundle (1,1) struct
    end
    d = bundle.descriptives;

    stat = { 'N'; 'Mean'; 'Std'; 'Skewness'; 'Kurtosis'; 'Min'; 'Max'; ...
             'ADF p-value'; 'PP p-value'; 'KPSS p-value'; ...
             'Jarque-Bera p-value'; 'ARCH-LM p-value'; 'BDS stat'; 'BDS p-value' };
    val  = [ d.n; d.mean; d.std; d.skewness; d.kurtosis; d.min; d.max; ...
             firstP(d.adf); firstP(d.pp); firstP(d.kpss); ...
             firstP(d.jb); firstP(d.archlm); d.bds.stat; d.bds.pValue ];

    T = table(stat, val, 'VariableNames', {'Statistic', 'Value'});
end


function p = firstP(s)
% Some diagnostic structs report a vector of p-values across lags; take the
% first. Single-test structs (ADF/PP/KPSS/JB) return their scalar.
    p = s.pValue;
    p = p(1);
end
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: PASS (3/3).

- [ ] **Step 5: Checkpoint** — `checkcode` clean.

---

### Task 8: T2 posteriors + T3 log-marginal-likelihood tables

**Files:**
- Create: `+report/tableT2posteriors.m`, `+report/tableT3logml.m`
- Test: add to `tests/tReportTables.m`

- [ ] **Step 1: Add the failing tests**

Append to `tests/tReportTables.m`'s `methods (Test)`:

```matlab
        function t2OneRowPerParameterPerModel(testCase)
            T = report.tableT2posteriors(testCase.bundle);
            testCase.verifyClass(T, 'table');
            testCase.verifyTrue(all(ismember( ...
                {'Model','Parameter','PosteriorMean','PosteriorStd'}, ...
                T.Properties.VariableNames)));
            srnRows = T(strcmp(T.Model, 'SVLTRECH-SRN'), :);
            testCase.verifyTrue(any(strcmp(srnRows.Parameter, 'v_z_1')));
        end

        function t3ListsLogMLPerSvModel(testCase)
            T = report.tableT3logml(testCase.bundle);
            testCase.verifyTrue(all(ismember({'Model','LogMarginalLik'}, ...
                T.Properties.VariableNames)));
            srn = T(strcmp(T.Model, 'SVLTRECH-SRN'), :);
            testCase.verifyTrue(isfinite(srn.LogMarginalLik));
        end
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: FAIL — the two builders do not exist.

- [ ] **Step 3: Implement both builders**

Create `+report/tableT2posteriors.m`:

```matlab
function T = tableT2posteriors(bundle)
% tableT2posteriors  T2: posterior mean + std of every parameter per model.
%   Long format (heterogeneous parameter sets across models). GARCH point
%   estimates carry NaN std. Reads bundle.models only.
    arguments
        bundle (1,1) struct
    end
    M = bundle.models;
    modelCol = {}; paramCol = {}; meanCol = []; stdCol = [];
    for m = 1:numel(M)
        names = M(m).paramNames;
        mu    = M(m).posteriorMean;
        sd    = M(m).posteriorStd;
        for p = 1:numel(names)
            modelCol{end+1, 1} = M(m).name;       %#ok<AGROW>
            paramCol{end+1, 1} = names{p};        %#ok<AGROW>
            meanCol(end+1, 1)  = mu(p);           %#ok<AGROW>
            if isempty(sd)
                stdCol(end+1, 1) = NaN;           %#ok<AGROW>
            else
                stdCol(end+1, 1) = sd(p);         %#ok<AGROW>
            end
        end
    end
    T = table(modelCol, paramCol, meanCol, stdCol, ...
        'VariableNames', {'Model','Parameter','PosteriorMean','PosteriorStd'});
end
```

Create `+report/tableT3logml.m`:

```matlab
function T = tableT3logml(bundle)
% tableT3logml  T3: log marginal likelihood per model (NaN for MLE GARCH).
%   Higher is better. Reads bundle.models only.
    arguments
        bundle (1,1) struct
    end
    M = bundle.models;
    model = {M.name}';
    logml = [M.logMarginalLik]';
    T = table(model, logml, 'VariableNames', {'Model', 'LogMarginalLik'});
end
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: PASS (5/5).

- [ ] **Step 5: Checkpoint** — `checkcode` clean.

---

### Task 9: T4 scores + T5 Diebold-Mariano + T6 MCS tables

**Files:**
- Create: `+report/tableT4scores.m`, `+report/tableT5dm.m`, `+report/tableT6mcs.m`
- Test: add to `tests/tReportTables.m`

- [ ] **Step 1: Add the failing tests**

Append to `tests/tReportTables.m`'s `methods (Test)`:

```matlab
        function t4ScoresOneRowPerModel(testCase)
            T = report.tableT4scores(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models));
            testCase.verifyTrue(all(ismember( ...
                {'Model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'}, ...
                T.Properties.VariableNames)));
        end

        function t5DmRowPerBaseline(testCase)
            T = report.tableT5dm(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models) - 1);
            testCase.verifyTrue(all(ismember({'Baseline','DMstat','pValue'}, ...
                T.Properties.VariableNames)));
        end

        function t6McsMembership(testCase)
            T = report.tableT6mcs(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models));
            testCase.verifyTrue(all(ismember({'Model','inMCS','pValue'}, ...
                T.Properties.VariableNames)));
            testCase.verifyTrue(islogical(T.inMCS) || isnumeric(T.inMCS));
        end
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: FAIL — the three builders do not exist.

- [ ] **Step 3: Implement the three builders**

Create `+report/tableT4scores.m`:

```matlab
function T = tableT4scores(bundle)
% tableT4scores  T4: the five predictive scores + QLIKE, one row per model.
%   Reads bundle.comparison.{modelNames,scores}.
    arguments
        bundle (1,1) struct
    end
    names = bundle.comparison.modelNames(:);
    S     = bundle.comparison.scores;
    T = table(names, [S.pps]', [S.qs1]', [S.qs5]', [S.mse]', [S.mae]', ...
        [S.r2log]', [S.qlike]', ...
        'VariableNames', {'Model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'});
end
```

Create `+report/tableT5dm.m`:

```matlab
function T = tableT5dm(bundle)
% tableT5dm  T5: Diebold-Mariano statistic + p-value, each baseline vs the
%   proposed model. Negative stat => baseline has higher loss (proposed wins).
%   Reads bundle.comparison.{dmBaselines,dmVsProposed}.
    arguments
        bundle (1,1) struct
    end
    base = bundle.comparison.dmBaselines(:);
    dm   = bundle.comparison.dmVsProposed;
    T = table(base, [dm.statistic]', [dm.pValue]', ...
        'VariableNames', {'Baseline','DMstat','pValue'});
end
```

Create `+report/tableT6mcs.m`:

```matlab
function T = tableT6mcs(bundle)
% tableT6mcs  T6: Model Confidence Set membership + MCS p-value per model.
%   Reads bundle.comparison.{modelNames,mcs}.
    arguments
        bundle (1,1) struct
    end
    names = bundle.comparison.modelNames(:);
    mcs   = bundle.comparison.mcs;
    T = table(names, logical(mcs.inSet(:)), mcs.pValues(:), ...
        'VariableNames', {'Model','inMCS','pValue'});
end
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: PASS (8/8).

- [ ] **Step 5: Checkpoint** — `checkcode` clean.

---

### Task 10: T7 residual diagnostics table

**Files:**
- Create: `+report/tableT7residuals.m`
- Test: add to `tests/tReportTables.m`

- [ ] **Step 1: Add the failing test**

Append to `tests/tReportTables.m`'s `methods (Test)`:

```matlab
        function t7ResidualMomentsPerModel(testCase)
            T = report.tableT7residuals(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models));
            testCase.verifyTrue(all(ismember( ...
                {'Model','Mean','Std','Skewness','Kurtosis','LBQ2_pValue'}, ...
                T.Properties.VariableNames)));
            testCase.verifyTrue(all(isfinite(T.Std)));
        end
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: FAIL — `report.tableT7residuals` does not exist.

- [ ] **Step 3: Implement `+report/tableT7residuals.m`**

```matlab
function T = tableT7residuals(bundle)
% tableT7residuals  T7: standardized-residual moments + Ljung-Box on squared
%   residuals (leftover ARCH) per model. Reads bundle.models(m).stdResid.
    arguments
        bundle (1,1) struct
    end
    M = bundle.models;
    name = {}; mu = []; sd = []; sk = []; ku = []; lbq2 = [];
    for m = 1:numel(M)
        e = M(m).stdResid(:);
        e = e(isfinite(e));
        rd = data.preprocess.residualDiagnostics(e);
        name{end+1,1} = M(m).name;            %#ok<AGROW>
        mu(end+1,1)   = rd.moments.mean;       %#ok<AGROW>
        sd(end+1,1)   = rd.moments.std;        %#ok<AGROW>
        sk(end+1,1)   = rd.moments.skewness;   %#ok<AGROW>
        ku(end+1,1)   = rd.moments.kurtosis;   %#ok<AGROW>
        p = rd.mcleodli.pValue;                % Ljung-Box on squared resids
        lbq2(end+1,1) = p(1);                  %#ok<AGROW>
    end
    T = table(name, mu, sd, sk, ku, lbq2, ...
        'VariableNames', {'Model','Mean','Std','Skewness','Kurtosis','LBQ2_pValue'});
end
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportTables.m')"`
Expected: PASS (9/9).

- [ ] **Step 5: Checkpoint — all table builders green; `checkcode` clean.**

---

### Task 11: F1 forecast bands + F2 QQ figures

**Files:**
- Create: `+report/withDefaults.m`, `+report/saveFigure.m`, `+report/figForecastBands.m`, `+report/figQQ.m`
- Test: `tests/tReportFigures.m`

- [ ] **Step 1: Write the failing test**

Create `tests/tReportFigures.m`:

```matlab
classdef tReportFigures < matlab.unittest.TestCase
% tReportFigures  F1-F4 builders emit non-empty PNG + PDF from a bundle.

    properties
        bundle
        tmp
    end

    methods (TestClassSetup)
        function setup(testCase)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, 'fixtures'));
            testCase.bundle = syntheticBundle();
            testCase.tmp = tempname; mkdir(testCase.tmp);
            testCase.addTeardown(@() rmdir(testCase.tmp, 's'));
        end
    end

    methods (Test)
        function f1ForecastBands(testCase)
            f = fullfile(testCase.tmp, 'F1');
            fig = report.figForecastBands(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end

        function f2QQ(testCase)
            f = fullfile(testCase.tmp, 'F2');
            fig = report.figQQ(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end
    end
end

function assertNonEmptyFiles(testCase, stem)
    for ext = ["png", "pdf"]
        p = stem + "." + ext;
        d = dir(p);
        testCase.verifyNotEmpty(d, sprintf('missing %s', p));
        testCase.verifyGreaterThan(d.bytes, 0);
    end
end
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportFigures.m')"`
Expected: FAIL — `report.figForecastBands` does not exist.

- [ ] **Step 3: Add the shared figure helpers**

Create `+report/withDefaults.m`:

```matlab
function opts = withDefaults(opts)
% withDefaults  Shared figure-option defaults for the report.fig* builders.
    if ~isfield(opts, 'Visible'); opts.Visible = 'off'; end
    if ~isfield(opts, 'file');    opts.file    = '';    end
end
```

Create `+report/saveFigure.m`:

```matlab
function saveFigure(fig, stem)
% saveFigure  Write a figure to <stem>.png and <stem>.pdf (no-op if stem empty).
    if isempty(stem); return; end
    d = fileparts(stem);
    if ~isempty(d) && ~isfolder(d); mkdir(d); end
    exportgraphics(fig, [stem '.png'], 'Resolution', 150);
    exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector');
end
```

- [ ] **Step 4: Implement `+report/figForecastBands.m`**

```matlab
function fig = figForecastBands(bundle, opts)
% figForecastBands  F1: OOS returns with 95% one-step-ahead bands from the
%   proposed model. opts.file (stem) -> writes <stem>.png + <stem>.pdf.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    names = bundle.comparison.modelNames;
    ix    = find(strcmp(names, bundle.comparison.proposedName), 1);
    rec   = bundle.models(ix);
    yTest = bundle.data.y(bundle.data.testIdx);
    nu    = rec.nu;
    scale = sqrt((nu - 2) / nu) .* sqrt(rec.varForecast);
    band  = tinv(0.975, nu) .* scale;

    fig = figure('Visible', opts.Visible);
    t = (1:numel(yTest))';
    fill([t; flipud(t)], [band; flipud(-band)], [0.85 0.9 1.0], ...
        'EdgeColor', 'none'); hold on;
    plot(t, yTest, 'k.', 'MarkerSize', 6);
    plot(t, band, 'b-', t, -band, 'b-');
    xlabel('out-of-sample t'); ylabel('return');
    title(sprintf('F1: 95%% one-step bands (%s)', rec.name), 'Interpreter', 'none');
    hold off;
    report.saveFigure(fig, opts.file);
end
```

- [ ] **Step 5: Implement `+report/figQQ.m`**

```matlab
function fig = figQQ(bundle, opts)
% figQQ  F2: QQ-plots of standardized residuals vs N(0,1), one panel per model.
%   opts.file (stem) -> writes <stem>.png + <stem>.pdf.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    M     = bundle.models;
    nM    = numel(M);
    nCols = min(3, nM);
    nRows = ceil(nM / nCols);

    fig = figure('Visible', opts.Visible);
    for m = 1:nM
        subplot(nRows, nCols, m);
        e = M(m).stdResid(:); e = e(isfinite(e));
        qqplot(e);
        title(M(m).name, 'Interpreter', 'none');
    end
    sgtitle('F2: standardized-residual QQ vs N(0,1)');
    report.saveFigure(fig, opts.file);
end
```

- [ ] **Step 6: Run to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportFigures.m')"`
Expected: PASS (2/2).

- [ ] **Step 7: Checkpoint** — `checkcode` clean on all new files.

---

### Task 12: F3 covariate posterior + F4 omega-state figures

**Files:**
- Create: `+report/figCovariatePosterior.m`, `+report/figOmegaState.m`
- Test: add to `tests/tReportFigures.m`

- [ ] **Step 1: Add the failing tests**

Append to `tests/tReportFigures.m`'s `methods (Test)`:

```matlab
        function f3CovariatePosterior(testCase)
            f = fullfile(testCase.tmp, 'F3');
            fig = report.figCovariatePosterior(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end

        function f4OmegaState(testCase)
            f = fullfile(testCase.tmp, 'F4');
            fig = report.figOmegaState(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportFigures.m')"`
Expected: FAIL — the two builders do not exist.

- [ ] **Step 3: Implement `+report/figCovariatePosterior.m`**

```matlab
function fig = figCovariatePosterior(bundle, opts)
% figCovariatePosterior  F3: kernel-density posteriors of the covariate
%   coefficients v_z for the proposed model. opts.file -> PNG + PDF.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    names = bundle.comparison.modelNames;
    ix    = find(strcmp(names, bundle.comparison.proposedName), 1);
    rec   = bundle.models(ix);
    cols  = find(startsWith(rec.paramNames, 'v_z_'));

    fig = figure('Visible', opts.Visible);
    hold on;
    leg = {};
    for c = cols
        [f, xi] = ksdensity(rec.theta(:, c));
        plot(xi, f, 'LineWidth', 1.2);
        leg{end+1} = rec.paramNames{c}; %#ok<AGROW>
    end
    xline(0, 'k--');
    if ~isempty(leg); legend(leg, 'Interpreter', 'none', 'Location', 'best'); end
    xlabel('coefficient value'); ylabel('posterior density');
    title(sprintf('F3: covariate-coefficient posteriors (%s)', rec.name), ...
        'Interpreter', 'none');
    hold off;
    report.saveFigure(fig, opts.file);
end
```

- [ ] **Step 4: Implement `+report/figOmegaState.m`**

```matlab
function fig = figOmegaState(bundle, opts)
% figOmegaState  F4: recurrent-state omega_t path for models that expose one
%   (the SRN flagship). Interpretability plot. opts.file -> PNG + PDF.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    M   = bundle.models;
    fig = figure('Visible', opts.Visible);
    hold on;
    leg = {};
    for m = 1:numel(M)
        w = M(m).omegaPath;
        if isempty(w); continue; end
        plot(1:numel(w), w, 'LineWidth', 1.0);
        leg{end+1} = M(m).name; %#ok<AGROW>
    end
    if isempty(leg)
        text(0.5, 0.5, 'no recurrent-state model', 'HorizontalAlignment', 'center');
    else
        legend(leg, 'Interpreter', 'none', 'Location', 'best');
    end
    xlabel('in-sample t'); ylabel('\omega_t');
    title('F4: recurrent-state path \omega_t');
    hold off;
    report.saveFigure(fig, opts.file);
end
```

- [ ] **Step 5: Run to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); runtests('tests/tReportFigures.m')"`
Expected: PASS (4/4).

- [ ] **Step 6: Checkpoint** — all figure builders green; `checkcode` clean.

---

### Task 13: `make_paper_tables.m` orchestrator

**Files:**
- Create: `scripts/make_paper_tables.m`
- Test: add an orchestrator test to `tests/tReportTables.m`

- [ ] **Step 1: Add the failing test**

Append to `tests/tReportTables.m`'s `methods (Test)`:

```matlab
        function orchestratorEmitsAllArtifacts(testCase)
            addpath('scripts');
            tmp = tempname; mkdir(tmp);
            testCase.addTeardown(@() rmdir(tmp, 's'));
            bundle = testCase.bundle;                       %#ok<NASGU>
            save(fullfile(tmp, 'bundle.mat'), 'bundle');

            make_paper_tables(tmp);

            for t = 1:7
                csv = fullfile(tmp, 'tables', sprintf('T%d.csv', t));
                tex = fullfile(tmp, 'tables', sprintf('T%d.tex', t));
                testCase.verifyTrue(isfile(csv), sprintf('missing %s', csv));
                testCase.verifyTrue(isfile(tex), sprintf('missing %s', tex));
            end
            for f = 1:4
                png = fullfile(tmp, 'figures', sprintf('F%d.png', f));
                testCase.verifyTrue(isfile(png), sprintf('missing %s', png));
            end
        end
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); addpath('scripts'); runtests('tests/tReportTables.m')"`
Expected: FAIL — `make_paper_tables` is not defined.

- [ ] **Step 3: Implement `scripts/make_paper_tables.m`**

```matlab
function make_paper_tables(resultDir)
% make_paper_tables  Build paper tables T1-T7 + figures F1-F4 from a bundle.
%
%   make_paper_tables(resultDir)
%
%   Loads <resultDir>/bundle.mat and writes:
%     <resultDir>/tables/T1..T7.{csv,tex}
%     <resultDir>/figures/F1..F4.{png,pdf}
%
%   The report layer is a pure consumer of the bundle (design spec §2/A);
%   no inference is re-run. See PROPOSED_METHODOLOGY.md §12.

    arguments
        resultDir (1,:) char
    end

    S = load(fullfile(resultDir, 'bundle.mat'), 'bundle');
    bundle = S.bundle;
    report.assertBundle(bundle);

    tDir = fullfile(resultDir, 'tables');
    fDir = fullfile(resultDir, 'figures');
    if ~isfolder(tDir); mkdir(tDir); end
    if ~isfolder(fDir); mkdir(fDir); end

    %% Tables (CSV + LaTeX). Columns: number, builder, lower-better, higher-better.
    specs = {
        1, @report.tableT1descriptives, {},                                  {}
        2, @report.tableT2posteriors,   {},                                  {}
        3, @report.tableT3logml,        {},                                  {'LogMarginalLik'}
        4, @report.tableT4scores,       {'PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'}, {}
        5, @report.tableT5dm,           {},                                  {}
        6, @report.tableT6mcs,          {},                                  {}
        7, @report.tableT7residuals,    {},                                  {}
    };
    for i = 1:size(specs, 1)
        n   = specs{i, 1};
        T   = specs{i, 2}(bundle);
        csv = fullfile(tDir, sprintf('T%d.csv', n));
        tex = fullfile(tDir, sprintf('T%d.tex', n));
        writetable(T, csv);
        report.latexTable(T, struct('lowerBetter', {specs{i,3}}, ...
            'higherBetter', {specs{i,4}}, ...
            'caption', sprintf('Table T%d', n), 'label', sprintf('tab:T%d', n), ...
            'file', tex));
        fprintf('[make_paper_tables] wrote %s + .tex\n', csv);
    end

    %% Figures (PNG + PDF).
    figs = {1, @report.figForecastBands; 2, @report.figQQ; ...
            3, @report.figCovariatePosterior; 4, @report.figOmegaState};
    for i = 1:size(figs, 1)
        n   = figs{i, 1};
        fig = figs{i, 2}(bundle, struct('file', fullfile(fDir, sprintf('F%d', n))));
        close(fig);
        fprintf('[make_paper_tables] wrote F%d.{png,pdf}\n', n);
    end
end
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); addpath('scripts'); runtests('tests/tReportTables.m')"`
Expected: PASS (10/10).

- [ ] **Step 5: Checkpoint — full suite green**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); addpath('scripts'); r = runtests('tests'); assert(all([r.Passed])); disp('ALL GREEN')"`
Expected: `ALL GREEN`.

---

### Task 14: Offline synthetic demonstration + completion record

**Files:**
- Create: `scripts/run_stage4_demo.m`
- Create: `docs/PHASE_6_COMPLETE.md`

Produces a real (synthetic-data) `results/stage4_demo/` bundle + full T1–T7 +
F1–F4 — concrete offline proof of the whole chain.

- [ ] **Step 1: Create `scripts/run_stage4_demo.m`**

```matlab
%% run_stage4_demo  Offline end-to-end demo of the Phase-6 reporting chain.
%  Synthetic VN-like series -> 6-model Stage-4 fit -> bundle -> tables + figures.
%  No network, small N/M/J. Run:
%    matlab -batch "addpath(pwd); addpath('scripts'); run('scripts/run_stage4_demo.m')"

utils.reproducibility(20260528);

K        = 2;
T        = 320;
splitIdx = 240;
thetaTrue = [-0.5, 0.95, 0.25, -0.4, 8, 0.05, 0.20, ...
             0.05, -0.05, 0.05, 0.10, -0.10, 0.10, 0.0];   % 12 + 2
Z   = randn(T, K);
gen = models.SVLTRECH('nCovariates', K, 'leverage', 'cholesky');
gen.setCovariates(Z);
[y, ~] = gen.simulate(thetaTrue, T);

cfg = struct();
cfg.model    = struct('nCovariates', K, 'leverage', 'cholesky');
cfg.smc      = struct('N', 300, 'M', 40, 'nSweeps', 3, 'proposalScale', 0.5, ...
                      'essThreshold', 0.5, 'targetEss', 0.5, 'verbose', false);
cfg.forecast = struct('J', 30);
cfg.eval     = struct('alphaQS', [0.01, 0.05], 'mcsB', 500);
cfg.particleFilter = struct('clipLogWeight', -50);
cfg.baseSeed = 20260528;

outDir = fullfile('results', 'stage4_demo');
out = experiments.runStage4(y, Z, splitIdx, cfg, ...
    struct('writeOutputs', true, 'outDir', outDir, 'expName', 'stage4_demo'));

make_paper_tables(outDir);

fprintf('\n[demo] models: %s\n', strjoin(out.modelNames, ', '));
for m = 1:numel(out.modelNames)
    s = out.scores(m);
    fprintf('  %-14s QLIKE=%.3f PPS=%.3f inMCS=%d\n', ...
        out.modelNames{m}, s.qlike, s.pps, out.mcs.inSet(m));
end
fprintf('[demo] tables + figures in %s\n', outDir);
```

- [ ] **Step 2: Run the demo**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); addpath('scripts'); run('scripts/run_stage4_demo.m')"`
Expected: prints 6 model score rows; `results/stage4_demo/{bundle.mat,scores.csv}`,
`results/stage4_demo/tables/T1..T7.{csv,tex}`, `results/stage4_demo/figures/F1..F4.{png,pdf}` all exist. Allow ~5–10 min (four small SMC fits).

- [ ] **Step 3: Inspect outputs**

Run: `ls -R results/stage4_demo` and open one `.tex` + one `.png` to confirm
well-formed content. Record the demo's QLIKE/PPS table **as found** — do not tune.

- [ ] **Step 4: Write `docs/PHASE_6_COMPLETE.md`**

Mirror `docs/PHASE_5_STAGE4_COMPLETE.md`'s style. Cover: files added (registry,
omegaPath, bundle-persisting runStage4, +report package, BDS, demo); the test
tally (`runtests('tests')` count, PASS/FAIL); the synthetic demo's table/figure
inventory + the as-found score table; explicit note that the **live VN-Index run
remains deferred** (no network) and that SMC parallelization, Stage 5, and the
prose draft are out of scope. List the exact connected-machine command to obtain
real numbers (`scripts/run_stage4_vietnam.m` — which must first persist a bundle
via the rewritten `runStage4` — then `make_paper_tables('results/stage4')`).

> Note: `scripts/run_stage4_vietnam.m` already calls `experiments.runStage4`
> with `writeOutputs=true`; after the Task-4 rewrite it now persists `bundle.mat`,
> so the only added step for the live deliverable is `make_paper_tables('results/stage4')`.

- [ ] **Step 5: Final checkpoint — full suite green**

Run: `/home/d0/MATLAB/R2026a/bin/matlab -batch "addpath(pwd); addpath('scripts'); r = runtests('tests'); assert(all([r.Passed])); fprintf('ALL GREEN: %d tests\n', numel(r))"`
Expected: `ALL GREEN: <N> tests`.

---

## Self-review

**Spec coverage** (against `docs/superpowers/specs/2026-05-28-phase6-reporting-figures-design.md`):
- §3.1 results-bundle schema + persistence → Task 4 (`bundle` assembly + `writeStage4Outputs` save) ✓
- §3.1 model-set expansion 4→6 via registry → Task 2 (`stage4Models`) + Task 4 (loop) ✓
- §3.1 table builders T1–T7 → Tasks 7–10 ✓
- §3.1 `make_paper_tables.m` (CSV+LaTeX) → Task 13 ✓
- §3.1 `latexTable` booktabs (bold-best, MCSE) → Task 6 ✓
- §3.1 figure builders F1–F4 (PNG+PDF) → Tasks 11–12 ✓
- §3.1 compact BDS for T1 → Task 1 ✓
- §3.1 tests for every unit + regression guard → every task's test + Task 4 Step 5 / Task 13 Step 5 ✓
- §3.1 offline synthetic demo → Task 14 ✓
- §4.1 bundle fields (.meta/.data/.descriptives/.models/.comparison) → Task 4 assembly + Task 5 fixture mirror ✓
- §4.2 file table (every listed file) → Tasks 1–13 ✓ (`assertBundle` Task 5; `withDefaults`/`saveFigure` Task 11)
- §6 validation anchors (shape contracts, bold-best, BDS sanity, figure emission, as-found) → Tasks 1/6/9/11/12/14 ✓
- §8 acceptance criteria 1–6 → Tasks 4/6–10/11–12/13/1/14 ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. The only narrative step is Task 14 Step 4 (writing a prose completion doc), which enumerates required content explicitly. ✓

**Type/signature consistency:**
- Bundle field names identical across writer (Task 4 `bundle.*`), fixture (Task 5), validator (`assertBundle`), and every builder (Tasks 7–13). ✓
- `report.fig*(bundle, opts)` with `opts.file` stem + shared `report.withDefaults`/`report.saveFigure` — consistent Tasks 11–13. ✓
- `emptyScore`/`emptyModelRecord` field sets defined Task 4; `out.scores` struct array consumed by `tableT4scores` (Task 9) matches; `bundle.models(m)` fields consumed by Tasks 7/8/10/11/12 match the record. ✓
- DM count = nModels−1 = 5 (Task 4 `dmVsProposed` over `others`; smoke-test Task 4 Step 1; `tableT5dm` Task 9). ✓
- `nu` at θ-index 5 for all SV models (verified in source) used in `fitOneModel` (`thetaBar(5)`). ✓
- `latexTable` opts (`lowerBetter`/`higherBetter`/`caption`/`label`/`file`) defined Task 6, used identically Task 13. ✓
- `models.SVLTRECH.omegaPath(theta,hPath,yPath)` defined Task 3, called in Task 4 `fitOneModel` via `ismethod` guard, plotted in Task 12. ✓
- `data.preprocess.bdsTest` signature (`out.stat/.pValue/.m/.eps`) defined Task 1, consumed in Task 4 `describeSeries` + Task 5 fixture + Task 7 T1. ✓

**Known residual risks (flagged, not placeholders):** `omegaPath` is implemented for the SRN flagship only; LSTM/GRU records carry empty `omegaPath` and F4 omits them (Task 12 handles the empty case). GARCH `posteriorStd` is NaN (MLE point estimate; `latexTable` renders NaN as `--`). These are intentional per the design spec.
