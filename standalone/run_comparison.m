function results = run_comparison(y, Z, splitIdx, opts)
% RUN_COMPARISON  Self-contained SV / GARCH volatility-model comparison.
% =====================================================================
% ONE FILE, NO DEPENDENCIES. This script reproduces the full Stage-4
% comparison from the SV-LT-RECH project without any +package or external
% file. Drop it anywhere on the MATLAB path and run it.
%
% It fits and out-of-sample-forecasts a symmetric grid of volatility models
% and ranks them by Diebold-Mariano tests and the Model Confidence Set:
%
%                      no deep learning          + deep learning (SRN/LSTM/GRU)
%   GARCH backbone     GARCH-t, GJR-t            GARCH-RECH-{SRN,LSTM,GRU}
%   SV    backbone     SV, SVM, SVLT             SVLTRECH-{SRN,LSTM,GRU}
%
%   (SVM = stochastic volatility IN MEAN; SVLT = SV + leverage + Student-t.)
%
% USAGE
% -----
%   run_comparison                         % built-in synthetic demo
%   results = run_comparison(y)            % your T-by-1 returns; 80/20 split, no covariates
%   results = run_comparison(y, Z)         % + T-by-K covariates in the variance eqn
%   results = run_comparison(y, Z, splitIdx)
%   results = run_comparison(y, Z, splitIdx, opts)
%
%   opts (struct, all optional):
%     .models     cellstr subset of model names to run (default: all 11)
%     .smcN       SMC parameter particles            (default 1000)
%     .smcM       PF latent particles                (default 200)
%     .nSweeps    RWM sweeps per anneal step         (default 8)
%     .forecastJ  thinned posterior draws for OOS    (default 100)
%     .meanForce  'auto' | 'zero' | 'arma'           (default 'auto')
%     .grRestarts GARCH-RECH MLE multistarts         (default 4)
%     .alphaQS    VaR levels for quantile score      (default [0.01 0.05])
%     .mcsB       MCS bootstrap reps                 (default 2000)
%     .seed       base RNG seed                       (default 20260516)
%     .verbose    print progress                      (default true)
%
% RETURNS  results struct: .modelNames .scores (table) .dm .mcs .forecasts
%          .meanEquation .config
%
% REQUIREMENTS
%   MATLAB R2026a (or recent). Statistics & ML Toolbox (distributions).
%   Econometrics Toolbox is used for the classic GARCH-t/GJR-t baselines and
%   the ARMA mean-equation gate; if it is absent those parts are skipped and
%   the mean equation falls back to zero-mean (SV + GARCH-RECH still run).
%   Optimization Toolbox is used for GARCH-RECH (falls back to fminsearch).
%
% This file is a faithful flattening of the tested +package codebase; the
% mathematics is copied verbatim from the unit-tested source. See the
% project test suite (runtests('tests')) for the authoritative checks.
% =====================================================================

    %% -------- argument handling + demo data --------
    if nargin < 4 || isempty(opts); opts = struct(); end
    if nargin < 1 || isempty(y)
        [y, Z, splitIdx] = demoData();
        fprintf('[run_comparison] No data supplied -> built-in synthetic demo.\n');
    else
        if nargin < 2; Z = []; end
        if nargin < 3 || isempty(splitIdx); splitIdx = floor(0.8 * numel(y)); end
    end

    cfg = defaultConfig(opts);
    rng(cfg.seed, 'threefry');

    y = y(:);
    T = numel(y);
    if isempty(Z); Z = zeros(T, 0); end
    K = size(Z, 2);
    assert(splitIdx > 10 && splitIdx < T, 'splitIdx out of range');

    hasEcon = ~isempty(ver('econ'));

    %% -------- mean equation (ARMA auto-gate) on residuals --------
    [r, meanInfo] = meanEquationLocal(y, splitIdx, cfg.meanForce, hasEcon);
    if cfg.verbose
        if meanInfo.usedARMA
            fprintf('[mean] ARMA(%d,%d) fitted (Ljung-Box p=%.3f).\n', ...
                meanInfo.p, meanInfo.q, meanInfo.ljungP);
        else
            fprintf('[mean] zero-mean (returns pass through).\n');
        end
    end

    testIdx = (splitIdx + 1 : T)';
    rTest   = r(testIdx);
    rvProxy = max(rTest .^ 2, 1e-8);
    rvSqrt  = sqrt(rvProxy);

    %% -------- model registry --------
    reg = modelRegistry();
    if isfield(cfg, 'models') && ~isempty(cfg.models)
        reg = reg(ismember({reg.name}, cfg.models));
    end
    if ~hasEcon
        keep = ~strcmp({reg.kind}, 'garch');     % classic GARCH needs Econ
        if any(~keep)
            warning('Econometrics Toolbox absent: skipping %s.', ...
                strjoin({reg(~keep).name}, ', '));
        end
        reg = reg(keep);
    end
    nModels    = numel(reg);
    modelNames = {reg.name};

    %% -------- fit + score every model --------
    scoreRows = cell(nModels, 8);
    qlAll     = zeros(numel(testIdx), nModels);
    forecasts = struct();
    for mIx = 1:nModels
        spec = reg(mIx);
        if cfg.verbose; fprintf('[fit] %-16s ...\n', spec.name); end
        rec = fitOne(spec, r, Z, splitIdx, K, cfg);
        [sc, ql] = scoreModelLocal(rec.varForecast, rec.logPredDensity, rec.nu, ...
                                   rTest, rvProxy, rvSqrt, cfg.alphaQS);
        qlAll(:, mIx) = ql;
        scoreRows(mIx, :) = {spec.name, sc.pps, sc.qs1, sc.qs5, sc.mse, sc.mae, sc.qlike};
        forecasts.(matlab.lang.makeValidName(spec.name)) = rec.varForecast;
    end

    scores = cell2table(scoreRows, 'VariableNames', ...
        {'model','PPS','QS1','QS5','MSE','MAE','QLIKE'});

    %% -------- Diebold-Mariano vs the flagship + Model Confidence Set --------
    propIx = find([reg.isProposed], 1);
    if isempty(propIx); propIx = find(strcmp(modelNames, 'SVLTRECH-SRN'), 1); end
    dm = struct('baseline', {}, 'statistic', {}, 'pValue', {});
    if ~isempty(propIx)
        for mIx = setdiff(1:nModels, propIx)
            d = dieboldMarianoLocal(qlAll(:, mIx), qlAll(:, propIx), 1);
            dm(end+1) = struct('baseline', modelNames{mIx}, ...
                'statistic', d.statistic, 'pValue', d.pValue); %#ok<AGROW>
        end
    end
    mcs = modelConfidenceSetLocal(qlAll, cfg.mcsB, cfg.seed);

    %% -------- report --------
    if cfg.verbose
        fprintf('\n===== Out-of-sample scores (lower is better) =====\n');
        disp(scores);
        inSet = modelNames(mcs.inSet);
        fprintf('Model Confidence Set (75%%): %s\n', strjoin(inSet, ', '));
        if ~isempty(propIx)
            fprintf('Diebold-Mariano vs %s (DM<0 => flagship better):\n', modelNames{propIx});
            for k = 1:numel(dm)
                fprintf('   %-16s DM=%+.2f  p=%.3f\n', dm(k).baseline, dm(k).statistic, dm(k).pValue);
            end
        end
    end

    results = struct('modelNames', {modelNames}, 'scores', scores, ...
                     'dm', dm, 'mcs', mcs, 'forecasts', forecasts, ...
                     'meanEquation', meanInfo, 'config', cfg, 'testIdx', testIdx);
end


% =====================================================================
%                          MODEL REGISTRY
% =====================================================================
function reg = modelRegistry()
    e = @(name, kind, gt, cl, build, prop) struct( ...
        'name', name, 'kind', kind, 'garchType', gt, 'cell', cl, ...
        'build', build, 'isProposed', prop);
    reg = e('GARCH-t',         'garch',     'garch', '',     [],                false);
    reg(end+1) = e('GJR-t',           'garch',     'gjr',   '',     [],                false);
    reg(end+1) = e('GARCH-RECH-SRN',  'garchrech', '',      'srn',  [],                false);
    reg(end+1) = e('GARCH-RECH-LSTM', 'garchrech', '',      'lstm', [],                false);
    reg(end+1) = e('GARCH-RECH-GRU',  'garchrech', '',      'gru',  [],                false);
    reg(end+1) = e('SV',              'sv', '', '', @(K,Z) build_SV(),               false);
    reg(end+1) = e('SVM',             'sv', '', '', @(K,Z) build_SVM(),              false);
    reg(end+1) = e('SVLT',            'sv', '', '', @(K,Z) build_SVLT('cholesky'),   false);
    reg(end+1) = e('SVLTRECH-SRN',    'sv', '', '', @(K,Z) build_RECH('srn', K,Z,'cholesky'),  true);
    reg(end+1) = e('SVLTRECH-LSTM',   'sv', '', '', @(K,Z) build_RECH('lstm',K,Z,'cholesky'),  false);
    reg(end+1) = e('SVLTRECH-GRU',    'sv', '', '', @(K,Z) build_RECH('gru', K,Z,'cholesky'),  false);
end


function rec = fitOne(spec, r, Z, splitIdx, K, cfg)
% Fit one model on residuals r; return varForecast, logPredDensity, nu on the
% test window (splitIdx+1:end).
    rTrain = r(1:splitIdx);
    switch spec.kind
        case 'garch'
            fit = fitGarchLocal(rTrain, spec.garchType);
            rf  = rollingForecastGarchLocal(fit, r, splitIdx);
            rec.nu = fit.nu; rec.varForecast = rf.sigma2Forecast;
            rec.logPredDensity = rf.logPredDensity;
        case 'garchrech'
            if K > 0; Ztr = Z(1:splitIdx, :); else; Ztr = []; end
            fit = fitGarchRECHLocal(rTrain, spec.cell, Ztr, cfg.grRestarts, cfg.verbose);
            rf  = rollingForecastRECHLocal(fit, r, Z, splitIdx);
            rec.nu = fit.nu; rec.varForecast = rf.sigma2Forecast;
            rec.logPredDensity = rf.logPredDensity;
        case 'sv'
            m   = spec.build(K, Z);          % model struct of function handles
            res = likelihoodAnnealLocal(m, rTrain, cfg);
            thetaBar = mean(res.theta, 1);
            nuIx = find(strcmp(m.paramNames, 'nu'), 1);
            if isempty(nuIx); rec.nu = Inf; else; rec.nu = thetaBar(nuIx); end
            fp = rollingPredictiveLocal(m, r, res.theta, splitIdx, cfg);
            rec.varForecast = fp.varForecast; rec.logPredDensity = fp.logPredDensity;
    end
end


% =====================================================================
%                 SV MODEL BUILDERS (struct of handles)
% =====================================================================
% Every model function takes the model struct `m` as its first argument so
% the engine can call m.fn(m, ...) without closures. theta layout matches the
% tested +models classes exactly.

function m = build_SV()
    m.name = 'SV';
    m.paramNames = {'mu','phi','sigma_eta'};
    m.np = 3;
    m.samplePrior = @sp_SV;       m.logPrior  = @lp_SV;
    m.initLatent  = @initLatentCommon;
    m.transition  = @tr_SV;       m.obsLogLik = @obs_gauss;
    m.initAux     = @aux_none;    m.updateAux = @aux_passthrough;
    m.Z = zeros(0,0); m.K = 0; m.couplingName = ''; m.cellType = '';
end

function m = build_SVM()
    m.name = 'SVM';
    m.paramNames = {'mu','phi','sigma_eta','alpha0','lambda'};
    m.np = 5;
    m.samplePrior = @sp_SVM;      m.logPrior  = @lp_SVM;
    m.initLatent  = @initLatentCommon;
    m.transition  = @tr_SV;       m.obsLogLik = @obs_svm;
    m.initAux     = @aux_none;    m.updateAux = @aux_passthrough;
    m.Z = zeros(0,0); m.K = 0; m.couplingName = ''; m.cellType = '';
end

function m = build_SVLT(leverage)
    m.name = 'SVLT';
    m.paramNames = {'mu','phi','sigma_eta','rho','nu'};
    m.np = 5;
    m.samplePrior = @sp_SVLT;     m.logPrior  = @lp_SVLT;
    m.initLatent  = @initLatentCommon;
    m.transition  = @tr_SVLT;     m.obsLogLik = @obs_scaledt;
    m.initAux     = @aux_lev;     m.updateAux = @aux_lev_update;
    m.Z = zeros(0,0); m.K = 0; m.couplingName = leverage; m.cellType = '';
end

function m = build_RECH(cellType, K, Z, leverage)
    m.name = sprintf('SVLTRECH-%s', upper(cellType));
    m.K = K; m.Z = Z; m.couplingName = leverage; m.cellType = cellType;
    nIn = 3 + K;
    switch cellType
        case 'srn'
            core = {'mu','phi','sigma_eta','rho','nu','beta_0','beta_1', ...
                    'v_h','v_r','v_omega'};
            cov  = arrayfun(@(k) sprintf('v_z_%d',k), 1:K, 'UniformOutput', false);
            m.paramNames = [core, cov, {'w_h','b'}];
            m.transition = @tr_RECH_srn;
        case 'lstm'
            m.paramNames = rechNames({'f','i','o','c'}, K);
            m.transition = @tr_RECH_lstm;
        case 'gru'
            m.paramNames = rechNames({'z','r','h'}, K);
            m.transition = @tr_RECH_gru;
    end
    m.np = numel(m.paramNames);
    m.samplePrior = @sp_RECH;   m.logPrior = @lp_RECH;
    m.initLatent  = @initLatentCommon;
    m.obsLogLik   = @obs_scaledt;
    if strcmp(cellType,'lstm'); m.initAux = @aux_rech_lstm; else; m.initAux = @aux_rech; end
    m.updateAux   = @aux_rech_update;
    m.nIn = nIn;
end

function names = rechNames(gates, K)
    core = {'mu','phi','sigma_eta','rho','nu','beta_0','beta_1'};
    inLabels = [{'h','r','omega'}, arrayfun(@(k) sprintf('z%d',k), 1:K, 'UniformOutput', false)];
    gn = {};
    for gi = 1:numel(gates)
        g = gates{gi};
        wN = cellfun(@(lbl) sprintf('W%s_%s', g, lbl), inLabels, 'UniformOutput', false);
        gn = [gn, wN, {sprintf('u%s',g), sprintf('b%s',g)}]; %#ok<AGROW>
    end
    names = [core, gn];
end


% ---- shared latent init (stationary) ----
function h0 = initLatentCommon(~, theta, n)
    mu = theta(1); phi = theta(2); se = theta(3);
    h0 = mu + se / sqrt(1 - phi^2) * randn(n, 1);
end

% ---- observation densities ----
function logp = obs_gauss(~, yT, hT, ~)
    logp = -0.5*log(2*pi) - 0.5*hT - 0.5*yT.^2 .* exp(-hT);
end
function logp = obs_svm(~, yT, hT, theta)
    meanT = theta(4) + theta(5) .* exp(hT);
    logp  = -0.5*log(2*pi) - 0.5*hT - 0.5*(yT - meanT).^2 .* exp(-hT);
end
function logp = obs_scaledt(~, yT, hT, theta)
    nu = theta(5);
    scaledSq = yT.^2 ./ (exp(hT) * (nu - 2));
    logp = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(pi*(nu-2)) ...
         - 0.5*hT - ((nu+1)/2)*log1p(scaledSq);
end

% ---- transitions ----
function [hNew, aux] = tr_SV(~, hOld, theta, ~, n, aux)
    mu = theta(1); phi = theta(2); se = theta(3);
    hNew = mu + phi*(hOld - mu) + se*randn(n,1);
end
function [hNew, aux] = tr_SVLT(m, hOld, theta, ~, n, aux)
    mu = theta(1); phi = theta(2); se = theta(3); rho = theta(4);
    eta  = leverageCouple(m.couplingName, rho, aux.epsilonPrev, n);
    hNew = mu + phi*(hOld - mu) + se*eta;
end
function [hNew, aux] = tr_RECH_srn(m, hOld, theta, t, n, aux)
    s  = unpack_srn(theta, m.K);
    x  = [hOld, aux.yPrev*ones(n,1), aux.omegaPrev];
    if m.K > 0; x = [x, repmat(m.Z(t-1,:), n, 1)]; end
    w  = struct('v', [s.v_h, s.v_r, s.v_omega, s.v_z(:)'], 'w_h', s.w_h, 'b', s.b);
    sNew = cell_srn(x, aux.sPrev, w);
    omegaNew = s.beta_0 + s.beta_1*sNew;
    eta  = leverageCouple(m.couplingName, s.rho, aux.epsilonPrev, n);
    hNew = s.mu + s.phi*(hOld - s.mu) + omegaNew + s.sigma_eta*eta;
    aux.sPrev = sNew; aux.omegaPrev = omegaNew;
end
function [hNew, aux] = tr_RECH_lstm(m, hOld, theta, t, n, aux)
    s  = unpack_lstm(theta, m.K);
    x  = [hOld, aux.yPrev*ones(n,1), aux.omegaPrev];
    if m.K > 0; x = [x, repmat(m.Z(t-1,:), n, 1)]; end
    [sNew, ctx] = cell_lstm(x, aux.sPrev, aux.cPrev, s.w);
    omegaNew = s.beta_0 + s.beta_1*sNew;
    eta  = leverageCouple(m.couplingName, s.rho, aux.epsilonPrev, n);
    hNew = s.mu + s.phi*(hOld - s.mu) + omegaNew + s.sigma_eta*eta;
    aux.sPrev = sNew; aux.cPrev = ctx.c; aux.omegaPrev = omegaNew;
end
function [hNew, aux] = tr_RECH_gru(m, hOld, theta, t, n, aux)
    s  = unpack_gru(theta, m.K);
    x  = [hOld, aux.yPrev*ones(n,1), aux.omegaPrev];
    if m.K > 0; x = [x, repmat(m.Z(t-1,:), n, 1)]; end
    sNew = cell_gru(x, aux.sPrev, s.w);
    omegaNew = s.beta_0 + s.beta_1*sNew;
    eta  = leverageCouple(m.couplingName, s.rho, aux.epsilonPrev, n);
    hNew = s.mu + s.phi*(hOld - s.mu) + omegaNew + s.sigma_eta*eta;
    aux.sPrev = sNew; aux.omegaPrev = omegaNew;
end

% ---- auxiliary state ----
function aux = aux_none(~, ~, n);          aux = struct('n', n); end
function aux = aux_passthrough(~, aux, ~, ~, ~, ~); end
function aux = aux_lev(~, ~, n);           aux = struct('epsilonPrev', zeros(n,1)); end
function aux = aux_lev_update(~, aux, hCurr, yCurr, ~, ~)
    aux.epsilonPrev = yCurr ./ exp(hCurr/2);
end
function aux = aux_rech(~, theta, n)
    aux = struct('epsilonPrev', zeros(n,1), 'yPrev', 0, ...
                 'sPrev', zeros(n,1), 'omegaPrev', theta(6)*ones(n,1));
end
function aux = aux_rech_lstm(~, theta, n)
    aux = struct('epsilonPrev', zeros(n,1), 'yPrev', 0, ...
                 'sPrev', zeros(n,1), 'cPrev', zeros(n,1), 'omegaPrev', theta(6)*ones(n,1));
end
function aux = aux_rech_update(~, aux, hCurr, yCurr, ~, ~)
    aux.epsilonPrev = yCurr ./ exp(hCurr/2);
    aux.yPrev       = yCurr;
end

% ---- prior samplers ----
function theta = sp_SV(~, n)
    theta = [10*randn(n,1), 2*betarnd(20,1.5,n,1)-1, abs(trnd(1,n,1))];
end
function theta = sp_SVM(~, n)
    theta = [10*randn(n,1), 2*betarnd(20,1.5,n,1)-1, abs(trnd(1,n,1)), randn(n,1), randn(n,1)];
end
function theta = sp_SVLT(~, n)
    theta = [10*randn(n,1), 2*betarnd(20,1.5,n,1)-1, abs(trnd(1,n,1)), 2*rand(n,1)-1, 2+exprnd(10,n,1)];
end
function theta = sp_RECH(m, n)
    mu=10*randn(n,1); phi=2*betarnd(20,1.5,n,1)-1; se=abs(trnd(1,n,1));
    rho=2*rand(n,1)-1; nu=2+exprnd(10,n,1); b0=0.5*rand(n,1); b1=0.5*rand(n,1);
    switch m.cellType
        case 'srn'
            vh=0.1*randn(n,1); vr=0.1*randn(n,1); vo=0.1*randn(n,1);
            vz=0.5*randn(n,m.K); wh=0.1*randn(n,1); bb=0.1*randn(n,1);
            theta = [mu,phi,se,rho,nu,b0,b1,vh,vr,vo,vz,wh,bb];
        otherwise
            nGate = ifelse(strcmp(m.cellType,'lstm'),4,3) * (m.nIn + 2);
            theta = [mu,phi,se,rho,nu,b0,b1, 0.1*randn(n,nGate)];
    end
end

% ---- priors (log densities) ----
function lp = lp_SV(~, theta)
    mu=theta(1); phi=theta(2); se=theta(3);
    if phi<=-1||phi>=1||se<=0; lp=-Inf; return; end
    lp = -0.5*(mu/10)^2 - log(10) - 0.5*log(2*pi);
    ph=(1+phi)/2; lp = lp + (20-1)*log(ph) + (1.5-1)*log(1-ph) - betaln(20,1.5) - log(2);
    lp = lp + log(2) - log(pi) - log(1+se^2);
end
function lp = lp_SVM(~, theta)
    lp = lp_SV([], theta(1:3));
    if ~isfinite(lp); return; end
    lp = lp - 0.5*theta(4)^2 - 0.5*log(2*pi) - 0.5*theta(5)^2 - 0.5*log(2*pi);
end
function lp = lp_SVt_core(theta)   % [mu phi se nu]
    nu = theta(4);
    if theta(2)<=-1||theta(2)>=1||theta(3)<=0||nu<=2; lp=-Inf; return; end
    lp = lp_SV([], theta(1:3)) + log(0.1) - 0.1*nu + 0.1*2;   % Exp(0.1) trunc nu>2
end
function lp = lp_SVLT(~, theta)
    if theta(4)<=-1 || theta(4)>=1; lp=-Inf; return; end
    lp = lp_SVt_core([theta(1:3), theta(5)]) - log(2);
end
function lp = lp_RECH(m, theta)
    K=m.K; phi=theta(2); se=theta(3); rho=theta(4); nu=theta(5); b0=theta(6); b1=theta(7);
    if phi<=-1||phi>=1||se<=0||nu<=2||rho<=-1||rho>=1||b0<0||b0>0.5||b1<0||b1>0.5
        lp=-Inf; return;
    end
    lp = lp_SVLT([], [theta(1:4), theta(5)]) - 2*log(0.5);
    switch m.cellType
        case 'srn'
            vh=theta(8); vr=theta(9); vo=theta(10);
            if K>0; vz=theta(11:10+K); else; vz=[]; end
            wh=theta(11+K); bb=theta(12+K);
            sdS=0.1; lp = lp - 0.5*(vh^2+vr^2+vo^2+wh^2+bb^2)/sdS^2 - 5*(log(sdS)+0.5*log(2*pi));
            sdC=0.5; lp = lp - 0.5*sum(vz.^2)/sdC^2 - K*(log(sdC)+0.5*log(2*pi));
        otherwise
            gateW = theta(8:end); sd=0.1; nW=numel(gateW);
            lp = lp - 0.5*sum(gateW.^2)/sd^2 - nW*(log(sd)+0.5*log(2*pi));
    end
end

% ---- RECH theta unpackers (mirror +models classes exactly) ----
function s = unpack_srn(theta, K)
    s.mu=theta(1); s.phi=theta(2); s.sigma_eta=theta(3); s.rho=theta(4); s.nu=theta(5);
    s.beta_0=theta(6); s.beta_1=theta(7); s.v_h=theta(8); s.v_r=theta(9); s.v_omega=theta(10);
    if K>0; s.v_z=theta(11:10+K); else; s.v_z=zeros(1,0); end
    s.w_h=theta(11+K); s.b=theta(12+K);
end
function s = unpack_lstm(theta, K)
    s = unpackHead(theta); nIn = 3+K; off=7; w=struct();
    for g = {'f','i','o','c'}
        gn=g{1};
        w.(['W_' gn])=theta(off+1:off+nIn); w.(['u_' gn])=theta(off+nIn+1); w.(['b_' gn])=theta(off+nIn+2);
        off=off+nIn+2;
    end
    s.w = w;
end
function s = unpack_gru(theta, K)
    s = unpackHead(theta); nIn = 3+K; off=7; w=struct();
    for g = {'z','r','h'}
        gn=g{1};
        w.(['W_' gn])=theta(off+1:off+nIn); w.(['u_' gn])=theta(off+nIn+1); w.(['b_' gn])=theta(off+nIn+2);
        off=off+nIn+2;
    end
    s.w = w;
end
function s = unpackHead(theta)
    s.mu=theta(1); s.phi=theta(2); s.sigma_eta=theta(3); s.rho=theta(4); s.nu=theta(5);
    s.beta_0=theta(6); s.beta_1=theta(7);
end


% =====================================================================
%                          RNN CELLS
% =====================================================================
function sNew = cell_srn(x, sPrev, w)
    sNew = max(x * w.v(:) + w.w_h * sPrev + w.b, 0);
end
function [sNew, ctx] = cell_lstm(x, sPrev, cPrev, w)
    f = stableSigmoid(x*w.W_f(:) + w.u_f*sPrev + w.b_f);
    i = stableSigmoid(x*w.W_i(:) + w.u_i*sPrev + w.b_i);
    o = stableSigmoid(x*w.W_o(:) + w.u_o*sPrev + w.b_o);
    g = tanh(x*w.W_c(:) + w.u_c*sPrev + w.b_c);
    cNew = f.*cPrev + i.*g;
    sNew = o.*tanh(cNew);
    ctx = struct('c', cNew);
end
function sNew = cell_gru(x, sPrev, w)
    zg = stableSigmoid(x*w.W_z(:) + w.u_z*sPrev + w.b_z);
    rg = stableSigmoid(x*w.W_r(:) + w.u_r*sPrev + w.b_r);
    g  = tanh(x*w.W_h(:) + w.u_h*(rg.*sPrev) + w.b_h);
    sNew = (1 - zg).*sPrev + zg.*g;
end
function y = stableSigmoid(z); y = 0.5*(1 + tanh(z/2)); end


% =====================================================================
%                      LEVERAGE COUPLINGS
% =====================================================================
function etaNew = leverageCouple(name, rho, epsPrev, n)
    switch name
        case 'ocsn'
            [wPrior, mComp, vComp] = ocsnComponents();
            u    = log(epsPrev.^2 + 1e-12);
            logW = -0.5*log(2*pi*vComp) - 0.5*(u - mComp).^2 ./ vComp + log(wPrior);
            logW = logW - max(logW, [], 2);
            W    = exp(logW); W = W ./ sum(W, 2);
            cdf  = cumsum(W, 2); r = rand(n,1);
            kIdx = min(sum(r > cdf, 2) + 1, numel(wPrior));
            rhoEff = rho ./ (1 + vComp(kIdx)');
            etaNew = rhoEff.*epsPrev + sqrt(1 - rhoEff.^2).*randn(n,1);
        otherwise   % 'cholesky'
            etaNew = rho*epsPrev + sqrt(1 - rho^2)*randn(n,1);
    end
end
function [w, m, v] = ocsnComponents()
    w = [0.00609,0.04775,0.13057,0.20674,0.22715,0.18842,0.12047,0.05591,0.01575,0.00115];
    m = [1.92677,1.34744,0.73504,0.02266,-0.85173,-1.97278,-3.46788,-5.55246,-8.68384,-14.65000];
    v = [0.11265,0.17788,0.26768,0.40611,0.62699,0.98583,1.57469,2.54498,4.16591,7.33342];
end


% =====================================================================
%             SMC / PARTICLE-FILTER ENGINE (verbatim translation)
% =====================================================================
function y = logsumexp(x, dim)
    if nargin < 2; dim = find(size(x) > 1, 1); if isempty(dim); dim = 1; end; end
    mx = max(x, [], dim); fin = isfinite(mx);
    y = mx + log(sum(exp(x - mx), dim));
    y(~fin) = mx(~fin);
end

function e = essVal(logW); e = exp(2*logsumexp(logW(:)) - logsumexp(2*logW(:))); end

function idx = resampleSystematic(logW)
    N = numel(logW); logW = logW - logsumexp(logW(:)); w = exp(logW(:));
    u = (rand() + (0:N-1)')/N; cumW = cumsum(w); cumW(end) = 1;
    idx = zeros(N,1); j = 1;
    for i = 1:N
        while u(i) > cumW(j); j = j + 1; end
        idx(i) = j;
    end
end

function da = adaptiveTemperature(logW, ll, aMax, targetEss)
    if essAt(logW, ll, aMax) >= targetEss; da = aMax; return; end
    lo = 0; hi = aMax;
    for it = 1:60
        mid = 0.5*(lo+hi);
        if essAt(logW, ll, mid) > targetEss; lo = mid; else; hi = mid; end
        if hi - lo < 1e-10; break; end
    end
    da = 0.5*(lo+hi);
end
function e = essAt(logW, ll, da); e = essVal(logW + da*ll); end

function [logLik, hFiltered, logPredDensity, varForecast] = bootstrapPF(m, y, theta, M, clip, want)
    T = numel(y);
    h = m.initLatent(m, theta, M);
    aux = m.initAux(m, theta, M);
    logW = -log(M)*ones(M,1); logLik = 0;
    hFiltered = zeros(T,1); logPredDensity = zeros(T,1); varForecast = zeros(T,1);
    for t = 1:T
        if t > 1; [h, aux] = m.transition(m, h, theta, t, M, aux); end
        inc = m.obsLogLik(m, y(t), h, theta);
        if clip > -Inf; inc = max(inc, clip); end
        aux = m.updateAux(m, aux, h, y(t), theta, t);
        li = logsumexp(logW + inc); logLik = logLik + li;
        if want
            logPredDensity(t) = li; varForecast(t) = exp(logsumexp(logW + h));
        end
        logW = logW + inc - li;
        hFiltered(t) = sum(exp(logW).*h);
        if t < T
            if -logsumexp(2*logW) < log(0.5*M)
                idx = resampleSystematic(logW); h = h(idx); aux = reindexAux(aux, idx);
                logW = -log(M)*ones(M,1);
            end
        end
    end
end
function aux = reindexAux(aux, idx)
    fn = fieldnames(aux); M = numel(idx);
    for k = 1:numel(fn)
        v = aux.(fn{k});
        if isnumeric(v) && isvector(v) && numel(v) == M; aux.(fn{k}) = v(idx); end
    end
end

function res = likelihoodAnnealLocal(m, y, cfg)
    N = cfg.smcN; K = m.np; clip = cfg.clipLogWeight;
    theta = m.samplePrior(m, N);
    logLik = zeros(N,1);
    for i = 1:N; logLik(i) = bootstrapPF(m, y, theta(i,:), cfg.smcM, clip, false); end
    logW = -log(N)*ones(N,1); a = 0; logZ = 0;
    while a < 1
        da = adaptiveTemperature(logW, logLik, 1-a, 0.5*N);
        aNew = a + da;
        inc = da*logLik; lz = logsumexp(logW + inc);
        logZ = logZ + lz; logW = logW + inc - lz;
        if essVal(logW) < 0.5*N
            idx = resampleSystematic(logW); theta = theta(idx,:); logLik = logLik(idx);
            logW = -log(N)*ones(N,1);
        end
        propCov = cfg.proposalScale^2 * cov(theta) + 1e-8*eye(K);
        [theta, logLik] = rwMetropolis(m, theta, logLik, y, aNew, propCov, cfg.smcM, clip, cfg.nSweeps);
        a = aNew;
    end
    res.theta = theta; res.logLik = logLik; res.logMarginalLik = logZ;
end

function [theta, logLik] = rwMetropolis(m, theta, logLik, y, aK, propCov, M, clip, nSweeps)
    [N, K] = size(theta); L = chol(propCov, 'lower');
    for sweep = 1:nSweeps
        for i = 1:N
            cur = theta(i,:); llCur = logLik(i); lpCur = m.logPrior(m, cur);
            prop = (cur' + L*randn(K,1))'; lpProp = m.logPrior(m, prop);
            if isinf(lpProp); continue; end
            llProp = bootstrapPF(m, y, prop, M, clip, false);
            if log(rand()) < (lpProp + aK*llProp) - (lpCur + aK*llCur)
                theta(i,:) = prop; logLik(i) = llProp;
            end
        end
    end
end

function out = rollingPredictiveLocal(m, y, thetaPost, splitIdx, cfg)
    Ttot = numel(y); N = size(thetaPost, 1); J = min(cfg.forecastJ, N);
    pick = unique(round(linspace(1, N, J))); thetaThin = thetaPost(pick, :); J = size(thetaThin,1);
    logPredAll = zeros(Ttot, J); varAll = zeros(Ttot, J);
    for j = 1:J
        [~, ~, lpd, vf] = bootstrapPF(m, y, thetaThin(j,:), cfg.smcM, cfg.clipLogWeight, true);
        logPredAll(:,j) = lpd; varAll(:,j) = vf;
    end
    mx = max(logPredAll, [], 2);
    logPredBMA = mx + log(sum(exp(logPredAll - mx), 2)) - log(J);
    varBMA = mean(varAll, 2);
    testIdx = (splitIdx+1:Ttot)';
    out.varForecast = varBMA(testIdx); out.logPredDensity = logPredBMA(testIdx);
end


% =====================================================================
%                      GARCH FAMILY (classic + RECH)
% =====================================================================
function fit = fitGarchLocal(y, type)
    switch type
        case 'garch'; Mdl = garch(1,1);
        case 'gjr';   Mdl = gjr(1,1);
    end
    Mdl.Distribution = 't';
    est = estimate(Mdl, y, 'Display', 'off');
    fit.type = type; fit.dist = 't';
    fit.omega = est.Constant;
    fit.alpha = cell2mat(est.ARCH); fit.beta = cell2mat(est.GARCH);
    if strcmp(type,'gjr'); fit.gamma = cell2mat(est.Leverage); else; fit.gamma = []; end
    fit.nu = est.Distribution.DoF;
    fit.condVar = infer(est, y);
end

function out = rollingForecastGarchLocal(fit, y, splitIdx)
    T = numel(y); omega = fit.omega; alpha = fit.alpha(1); beta = fit.beta(1);
    if isempty(fit.gamma); gamma = 0; else; gamma = fit.gamma(1); end
    s2 = zeros(T,1); s2(1) = var(y(1:splitIdx));
    for t = 2:T
        rl = y(t-1);
        s2(t) = omega + alpha*rl^2 + gamma*rl^2*(rl<0) + beta*s2(t-1);
    end
    testIdx = (splitIdx+1:T)'; s2T = s2(testIdx); yT = y(testIdx);
    z = yT./sqrt(s2T);
    out.sigma2Forecast = s2T;
    out.logPredDensity = -0.5*log(s2T) + logStdT(z, fit.nu);
end

function fit = fitGarchRECHLocal(y, cellType, Z, restarts, verbose)
    T = numel(y); K = size(Z,2); d = 3 + K; nP = grParamCount(cellType, d);
    obj = @(p) grNegLogLik(p, y, Z, cellType, d);
    bestNll = inf; bestP = [];
    for r = 1:restarts
        p0 = grInit(nP, cellType, d, r);
        [pHat, nll] = grMinimize(obj, p0);
        if verbose; fprintf('   [GARCH-RECH:%s] start %d NLL=%.3f\n', cellType, r, nll); end
        if isfinite(nll) && nll < bestNll; bestNll = nll; bestP = pHat; end
    end
    P = grUnpack(bestP, cellType, d);
    [s2, omegaPath] = rechCondVar(P, y, Z, cellType, var(y));
    fit.cell = cellType; fit.nCovariates = K; fit.seedVar = var(y);
    fit.alpha = P.alpha; fit.beta = P.beta; fit.beta0 = P.beta0; fit.beta1 = P.beta1;
    fit.cellWeights = P.cellWeights; fit.nu = P.nu; fit.dist = 't';
    fit.condVar = s2; fit.omegaPath = omegaPath; fit.logLik = -bestNll;
end

function out = rollingForecastRECHLocal(fit, y, Z, splitIdx)
    P = struct('alpha',fit.alpha,'beta',fit.beta,'beta0',fit.beta0,'beta1',fit.beta1, ...
               'cellWeights', fit.cellWeights);
    s2 = rechCondVar(P, y, Z, fit.cell, fit.seedVar);
    testIdx = (splitIdx+1:numel(y))'; s2T = s2(testIdx); yT = y(testIdx);
    z = yT./sqrt(s2T);
    out.sigma2Forecast = s2T;
    out.logPredDensity = -0.5*log(s2T) + logStdT(z, fit.nu);
end

function [s2, omegaPath] = rechCondVar(P, y, Z, cellType, seedVar)
    T = numel(y); s2 = zeros(T,1); omegaPath = zeros(T,1);
    s2(1) = max(seedVar, 1e-8); omegaPath(1) = P.beta0;
    sPrev = 0; cPrev = 0; omegaPrev = P.beta0;
    emptyZ = isempty(Z);
    for t = 2:T
        if emptyZ; x = [s2(t-1), y(t-1), omegaPrev]; else; x = [s2(t-1), y(t-1), omegaPrev, Z(t-1,:)]; end
        switch cellType
            case 'srn';  sNew = cell_srn(x, sPrev, P.cellWeights);
            case 'lstm'; [sNew, ctx] = cell_lstm(x, sPrev, cPrev, P.cellWeights); cPrev = ctx.c;
            case 'gru';  sNew = cell_gru(x, sPrev, P.cellWeights);
        end
        omegaT = P.beta0 + P.beta1*sNew;
        s2(t) = max(omegaT + P.alpha*y(t-1)^2 + P.beta*s2(t-1), 1e-8);
        omegaPath(t) = omegaT; sPrev = sNew; omegaPrev = omegaT;
    end
end

function nll = grNegLogLik(p, y, Z, cellType, d)
    P = grUnpack(p, cellType, d);
    s2 = rechCondVar(P, y, Z, cellType, var(y));
    if any(~isfinite(s2)) || any(s2 <= 0); nll = inf; return; end
    z = y./sqrt(s2); ll = -0.5*log(s2) + logStdT(z, P.nu);
    nll = -sum(ll); if ~isfinite(nll); nll = inf; end
end

function P = grUnpack(p, cellType, d)
    tau = stableSigmoid(p(1)); psi = stableSigmoid(p(2));
    P.alpha = tau*psi; P.beta = tau*(1-psi);
    P.beta0 = grSoftplus(p(3)); P.beta1 = p(4); idx = 5;
    switch cellType
        case 'srn'
            v = p(idx:idx+d-1); idx = idx+d; w_h = p(idx); idx = idx+1; b = p(idx); idx = idx+1;
            P.cellWeights = struct('v', v(:)', 'w_h', w_h, 'b', b);
        case 'lstm'
            W = struct();
            for g = {'f','i','o','c'}
                gn=g{1}; W.(['W_' gn])=p(idx:idx+d-1); idx=idx+d;
                W.(['u_' gn])=p(idx); idx=idx+1; W.(['b_' gn])=p(idx); idx=idx+1;
            end
            P.cellWeights = W;
        case 'gru'
            W = struct();
            for g = {'z','r','h'}
                gn=g{1}; W.(['W_' gn])=p(idx:idx+d-1); idx=idx+d;
                W.(['u_' gn])=p(idx); idx=idx+1; W.(['b_' gn])=p(idx); idx=idx+1;
            end
            P.cellWeights = W;
    end
    P.nu = 2 + grSoftplus(p(idx));
end

function nP = grParamCount(cellType, d)
    switch cellType
        case 'srn';  nP = 4 + (d+2);
        case 'lstm'; nP = 4 + 4*(d+2);
        case 'gru';  nP = 4 + 3*(d+2);
    end
    nP = nP + 1;   % nu
end

function p0 = grInit(nP, cellType, d, restart) %#ok<INUSD>
    p0 = 0.05*randn(nP,1);
    p0(1) = 2.0 + 0.3*randn(); p0(2) = -1.0 + 0.3*randn();
    p0(3) = grInvSoftplus(0.05 + 0.1*rand()); p0(4) = 0.1*randn();
    p0(end) = grInvSoftplus(4 + 4*rand());
    if restart > 1; p0(3:end) = p0(3:end) + 0.25*randn(nP-2,1); end
end

function [pHat, fval] = grMinimize(obj, p0)
    if exist('fminunc','file')
        try
            o = optimoptions('fminunc','Algorithm','quasi-newton','Display','off', ...
                'MaxFunctionEvaluations',4000,'MaxIterations',4000);
            [pHat, fval] = fminunc(obj, p0, o); return
        catch
        end
    end
    o = optimset('Display','off','MaxFunEvals',4000,'MaxIter',4000);
    [pHat, fval] = fminsearch(obj, p0, o);
end

function y = grSoftplus(z); y = log1p(exp(-abs(z))) + max(z,0); end
function z = grInvSoftplus(y); z = log(expm1(max(y,1e-8))); end

function lp = logStdT(z, nu)
    lp = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log((nu-2)*pi) ...
       - ((nu+1)/2)*log1p(z.^2/(nu-2));
end


% =====================================================================
%                       EVALUATION METRICS
% =====================================================================
function [sc, qlSeries] = scoreModelLocal(varF, lpd, nu, yTest, rvProxy, rvSqrt, alphaQS)
    vHat = sqrt(varF);
    if isinf(nu)
        z1 = norminv(alphaQS(1)); z2 = norminv(alphaQS(2));
    else
        ts = sqrt((nu-2)/nu); z1 = tinv(alphaQS(1),nu)*ts; z2 = tinv(alphaQS(2),nu)*ts;
    end
    sc.pps = -mean(lpd);
    sc.qs1 = qScore(yTest, vHat*z1, alphaQS(1));
    sc.qs5 = qScore(yTest, vHat*z2, alphaQS(2));
    sc.mse = mean((rvSqrt - vHat).^2); sc.mae = mean(abs(rvSqrt - vHat));
    sc.r2log = mean(log(rvProxy ./ vHat.^2).^2);
    sc.qlike = mean(log(varF) + rvProxy ./ varF);
    qlSeries = log(varF) + rvProxy ./ varF;
end
function s = qScore(y, q, alpha); s = mean((alpha - double(y <= q)).*(y - q)); end

function out = dieboldMarianoLocal(loss1, loss2, h)
    T = numel(loss1); d = loss1 - loss2; dbar = mean(d); dc = d - dbar;
    bw = max(h-1, floor(T^(1/3))); s = mean(dc.^2);
    for k = 1:bw
        w = 1 - k/(bw+1); s = s + 2*w*mean(dc(1:end-k).*dc(k+1:end));
    end
    out.statistic = dbar/sqrt(max(s/T, 1e-12));
    out.pValue = 2*(1 - normcdf(abs(out.statistic)));
end

function out = modelConfidenceSetLocal(losses, B, seed)
    [T, K] = size(losses);
    if K < 2; out.inSet = true(K,1); out.pValues = ones(K,1); return; end
    rs = rng(); c = onCleanup(@() rng(rs)); rng(seed, 'threefry');
    L0 = round(sqrt(T)); idxB = statBootIdx(T, B, max(2,L0));
    active = true(K,1); pV = zeros(K,1);
    while sum(active) > 1
        id = find(active); Lj = losses(:, id);
        mL = mean(Lj,1); D = mL.' - mL;
        vD = zeros(numel(id));
        for b = 1:B
            mLb = mean(Lj(idxB(:,b),:),1); Db = mLb.' - mLb; vD = vD + (Db - D).^2;
        end
        vD = vD/B; vD(vD<1e-12)=1e-12; sdD = sqrt(vD);
        tvec = max(D./sdD, [], 2); tStat = max(tvec);
        tBoot = zeros(B,1);
        for b = 1:B
            mLb = mean(Lj(idxB(:,b),:),1); Db = mLb.' - mLb; cen = Db - D;
            tBoot(b) = max(max(cen./sdD, [], 2));
        end
        pVal = mean(tBoot >= tStat); pV(id) = max(pV(id), pVal);
        if pVal > 0.25; break; end
        [~, worst] = max(tvec); active(id(worst)) = false;
    end
    out.inSet = active; out.pValues = pV;
end
function I = statBootIdx(T, B, L)
    p = 1/L; I = zeros(T,B);
    for b = 1:B
        idx = randi(T); I(1,b) = idx;
        for t = 2:T
            if rand() < p; idx = randi(T); else; idx = idx+1; if idx>T; idx=1; end; end
            I(t,b) = idx;
        end
    end
end


% =====================================================================
%                        MEAN EQUATION (ARMA gate)
% =====================================================================
function [resid, info] = meanEquationLocal(y, splitIdx, force, hasEcon)
    info = struct('usedARMA', false, 'p', 0, 'q', 0, 'bic', NaN, 'ljungP', NaN, 'force', force);
    yTr = y(1:splitIdx);
    if ~hasEcon || strcmp(force, 'zero')
        resid = y; return;
    end
    [ljH, ljP] = lbqtest(yTr, 'Lags', 10, 'Alpha', 0.05);
    info.ljungP = ljP(end);
    useArma = (ljH(end) && ~strcmp(force,'zero')) || strcmp(force,'arma');
    if ~useArma; resid = y; return; end
    bestBic = inf; bestMdl = []; bestPQ = [0 0];
    for p = 0:3
        for q = 0:3
            if p==0 && q==0; continue; end
            try
                mdl = arima('ARLags',1:p,'MALags',1:q,'Constant',NaN);
                [est, ~, logL] = estimate(mdl, yTr, 'Display', 'off');
                bic = -2*logL + (p+q+2)*log(numel(yTr));
                if bic < bestBic; bestBic = bic; bestMdl = est; bestPQ = [p q]; end
            catch
            end
        end
    end
    if isempty(bestMdl); resid = y; return; end
    resid = infer(bestMdl, y);
    info.usedARMA = true; info.p = bestPQ(1); info.q = bestPQ(2); info.bic = bestBic;
end


% =====================================================================
%                       CONFIG + DEMO DATA + UTILS
% =====================================================================
function cfg = defaultConfig(opts)
    d.smcN = 1000; d.smcM = 200; d.nSweeps = 8; d.proposalScale = 0.5;
    d.clipLogWeight = -50; d.forecastJ = 100; d.meanForce = 'auto';
    d.grRestarts = 4; d.alphaQS = [0.01 0.05]; d.mcsB = 2000;
    d.seed = 20260516; d.verbose = true;
    cfg = d;
    fn = fieldnames(opts);
    for k = 1:numel(fn); cfg.(fn{k}) = opts.(fn{k}); end
end

function [y, Z, splitIdx] = demoData()
% Synthetic series from an SVLTRECH-SRN truth (K=2 covariates), T=400.
    rng(20260516, 'threefry');
    T = 400; K = 2; Z = randn(T, K);
    theta = [-0.5, 0.95, 0.25, -0.4, 8, 0.05, 0.20, 0.05, -0.05, 0.05, 0.10, -0.10, 0.10, 0.0];
    h = zeros(T,1); y = zeros(T,1); s = unpack_srn(theta, K); sc = sqrt((s.nu-2)/s.nu);
    h(1) = s.mu + s.sigma_eta/sqrt(1-s.phi^2)*randn(); y(1) = exp(h(1)/2)*sc*trnd(s.nu);
    sPrev = 0; omegaPrev = s.beta_0; epsPrev = y(1)/exp(h(1)/2); yPrev = y(1);
    for t = 2:T
        x = [h(t-1), yPrev, omegaPrev, Z(t-1,:)];
        w = struct('v',[s.v_h,s.v_r,s.v_omega,s.v_z(:)'],'w_h',s.w_h,'b',s.b);
        sNew = cell_srn(x, sPrev, w); omegaNew = s.beta_0 + s.beta_1*sNew;
        eta = leverageCouple('cholesky', s.rho, epsPrev, 1);
        h(t) = s.mu + s.phi*(h(t-1)-s.mu) + omegaNew + s.sigma_eta*eta;
        y(t) = exp(h(t)/2)*sc*trnd(s.nu);
        sPrev = sNew; omegaPrev = omegaNew; epsPrev = y(t)/exp(h(t)/2); yPrev = y(t);
    end
    splitIdx = 320;
end

function out = ifelse(cond, a, b); if cond; out = a; else; out = b; end; end
