function out = model_SVLTRECH(y, Z, splitIdx, cellType, opts)
% MODEL_SVLTRECH  SV deep-learning flagship: SV + leverage + Student-t with an
% RNN-augmented long-term volatility component (SVLTRECH), fit by Bayesian SMC.
% =====================================================================
% ONE FILE, NO DEPENDENCIES. This is the flagship "SVLTRECH" model extracted
% verbatim from the reviewed canonical run_comparison.m. Drop it anywhere on
% the MATLAB path and run it.
%
% MODEL
% -----
% The log-variance h_t follows a stationary AR(1) stochastic-volatility (SV)
% backbone, augmented by a recurrent long-term volatility component:
%
%       h_t = mu + phi*(h_{t-1} - mu) + omega_t + sigma_eta * eta_t
%
% where omega_t = beta_0 + beta_1 * s_t is produced by an RNN cell s_t
% (SRN / LSTM / GRU). The cell is driven by lagged log-variance, the lagged
% return, the lagged omega, and any exogenous covariates Z. This is the SV
% analogue of RECH (Recurrent Conditional Heteroskedasticity): the SV
% backbone REPLACES RECH's GARCH backbone, so the recurrent network feeds a
% latent-volatility state equation rather than a deterministic GARCH variance.
%
% Leverage enters through eta_t, coupled to the previous standardized return
% epsilon_{t-1} (here a Cholesky coupling with correlation rho). The
% observation density is a scaled Student-t with nu degrees of freedom:
%
%       y_t = exp(h_t / 2) * sqrt((nu-2)/nu) * t_nu .
%
% cellType selects the recurrent cell:
%   'srn'  (default, the flagship)  | 'lstm' | 'gru'.
%
% Estimation is fully Bayesian via Sequential Monte Carlo (SMC) with
% likelihood annealing; each parameter particle's likelihood is evaluated by a
% bootstrap particle filter over the latent volatility path, and RWM sweeps
% refresh the particles at each temperature step. Out-of-sample forecasts use
% Bayesian model averaging over thinned posterior draws.
%
% USAGE
% -----
%   model_SVLTRECH                                  % built-in synthetic demo (SRN)
%   out = model_SVLTRECH(y)                         % your T-by-1 returns; 80/20 split, no covariates
%   out = model_SVLTRECH(y, Z)                      % + T-by-K covariates in the variance eqn
%   out = model_SVLTRECH(y, Z, splitIdx)
%   out = model_SVLTRECH(y, Z, splitIdx, cellType)  % 'srn'|'lstm'|'gru'
%   out = model_SVLTRECH(y, Z, splitIdx, cellType, opts)
%
%   opts (struct, all optional): see defaultConfig below. The main runtime
%   knobs are cfg.smcN (parameter particles) and cfg.smcM (latent particles);
%   RUNTIME SCALES roughly linearly with cfg.smcN * cfg.smcM (and with
%   cfg.nSweeps and cfg.forecastJ), so lower them for quick smoke tests.
%
% RETURNS  out struct: .cell .scores .posteriorMean .paramNames .varForecast
%          .nu .logMarginalLik .meanEquation .testIdx
%
% This file is a faithful flattening of the tested +package codebase; the
% mathematics is copied verbatim from the unit-tested source.
% =====================================================================

    %% -------- argument handling + demo data --------
    if nargin < 5 || isempty(opts); opts = struct(); end
    if nargin < 4 || isempty(cellType); cellType = 'srn'; end
    if nargin < 1 || isempty(y)
        [y, Z, splitIdx] = demoData();
        fprintf('[model_SVLTRECH] No data supplied -> built-in synthetic demo.\n');
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

    %% -------- build + fit the SVLTRECH model --------
    m = build_RECH(cellType, K, Z, 'cholesky');
    if cfg.verbose; fprintf('[fit] %-16s ...\n', m.name); end
    res = likelihoodAnnealLocal(m, r(1:splitIdx), cfg);

    thetaBar = mean(res.theta, 1);
    nuIx = find(strcmp(m.paramNames, 'nu'), 1);
    if isempty(nuIx); nu = Inf; else; nu = thetaBar(nuIx); end

    fp = rollingPredictiveLocal(m, r, res.theta, splitIdx, cfg);

    %% -------- score on the test window --------
    [sc, ~] = scoreModelLocal(fp.varForecast, fp.logPredDensity, nu, ...
                              rTest, rvProxy, rvSqrt, cfg.alphaQS);

    %% -------- report --------
    if cfg.verbose
        fprintf('\n===== %s out-of-sample scores (lower is better) =====\n', m.name);
        fprintf('  PPS   = %.4f\n', sc.pps);
        fprintf('  QS1   = %.4f\n', sc.qs1);
        fprintf('  QS5   = %.4f\n', sc.qs5);
        fprintf('  MSE   = %.4f\n', sc.mse);
        fprintf('  MAE   = %.4f\n', sc.mae);
        fprintf('  QLIKE = %.4f\n', sc.qlike);
        fprintf('  nu    = %.3f   logMarginalLik = %.3f\n', nu, res.logMarginalLik);
    end

    out = struct('cell', cellType, 'scores', sc, ...
                 'posteriorMean', thetaBar, 'paramNames', {m.paramNames}, ...
                 'varForecast', fp.varForecast, 'nu', nu, ...
                 'logMarginalLik', res.logMarginalLik, ...
                 'meanEquation', meanInfo, 'testIdx', testIdx);
end


% =====================================================================
%                 SV MODEL BUILDER (struct of handles)
% =====================================================================
% Every model function takes the model struct `m` as its first argument so
% the engine can call m.fn(m, ...) without closures. theta layout matches the
% tested +models classes exactly.

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

% ---- observation density ----
function logp = obs_scaledt(~, yT, hT, theta)
    nu = theta(5);
    scaledSq = yT.^2 ./ (exp(hT) * (nu - 2));
    logp = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(pi*(nu-2)) ...
         - 0.5*hT - ((nu+1)/2)*log1p(scaledSq);
end

% ---- transitions ----
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

% ---- prior sampler ----
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
    m = build_RECH('srn', K, Z, 'cholesky');
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
