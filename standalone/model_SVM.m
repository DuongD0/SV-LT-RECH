function out = model_SVM(y, Z, splitIdx, opts)
% MODEL_SVM  Self-contained stochastic-volatility-IN-MEAN (SV-M) model.
% =====================================================================
% ONE FILE, NO DEPENDENCIES. Fits the SV-in-mean model of Koopman & Hol
% Uspensky (2002, J. Applied Econometrics) via the SMC particle-filter
% engine, produces out-of-sample variance forecasts and one-step
% predictive densities, and scores them.
%
% MODEL (SV-in-mean / SVM)
% ------------------------
%   y_t      = alpha0 + lambda * exp(h_t) + exp(h_t/2) * eps_t,  eps_t ~ N(0,1)
%   h_t - mu = phi*(h_{t-1} - mu) + sigma_eta * u_t,             u_t ~ N(0,1)
%
% theta = [mu, phi, sigma_eta, alpha0, lambda].
%
% THE VARIANCE-IN-MEAN TERM  lambda * exp(h_t)
% --------------------------------------------
% The conditional mean of the return is shifted by lambda times the
% *conditional variance* exp(h_t). This is the empirical risk-return
% trade-off (intertemporal CAPM / "feedback" effect): when latent
% volatility exp(h_t) is high, the expected return moves by lambda *
% exp(h_t). A positive lambda encodes the textbook hypothesis that
% investors demand higher expected returns to bear higher variance risk
% (a positive volatility risk premium); a negative lambda captures the
% "volatility feedback" channel in which anticipated increases in
% variance depress contemporaneous prices. alpha0 is the baseline mean.
% Because exp(h_t) is itself a stochastic latent state, lambda is
% identified jointly with the volatility dynamics inside the particle
% filter rather than from a separate variance proxy.
%
% USAGE
% -----
%   model_SVM                         % built-in synthetic demo
%   out = model_SVM(y)                % your T-by-1 returns; 80/20 split
%   out = model_SVM(y, Z)             % Z is ignored by SVM (variance has no covariates)
%   out = model_SVM(y, Z, splitIdx)
%   out = model_SVM(y, Z, splitIdx, opts)
%
% RETURNS  out struct: .scores .posteriorMean .paramNames .varForecast
%          .nu .logMarginalLik .meanEquation .testIdx
%
% The SMC engine, mean equation, scoring and demo data are copied verbatim
% from the reviewed canonical run_comparison.m (this file is a standalone
% flattening for the SVM model only). Statistics & ML Toolbox is used for
% distributions; the Econometrics Toolbox (if present) enables the ARMA
% mean-equation gate, otherwise it falls back to zero-mean.
% =====================================================================

    %% -------- argument handling + demo data --------
    if nargin < 1 || isempty(y)
        [y, Z, splitIdx] = demoData();
        fprintf('[model_SVM] No data supplied -> built-in synthetic demo.\n');
    else
        if nargin < 2; Z = []; end
        if nargin < 3 || isempty(splitIdx); splitIdx = floor(0.8 * numel(y)); end
    end
    if nargin < 4 || isempty(opts); opts = struct(); end

    cfg = defaultConfig(opts);
    rng(cfg.seed, 'threefry');

    y = y(:);
    T = numel(y);
    assert(splitIdx > 10 && splitIdx < T, 'splitIdx out of range');

    %% -------- mean equation (ARMA auto-gate) on residuals --------
    [r, meanInfo] = meanEquationLocal(y, splitIdx, cfg.meanForce, ~isempty(ver('econ')));
    if cfg.verbose
        if meanInfo.usedARMA
            fprintf('[mean] ARMA(%d,%d) fitted (Ljung-Box p=%.3f).\n', ...
                meanInfo.p, meanInfo.q, meanInfo.ljungP);
        else
            fprintf('[mean] zero-mean (returns pass through).\n');
        end
    end

    %% -------- build model + SMC fit on the training window --------
    m   = build_SVM();
    res = likelihoodAnnealLocal(m, r(1:splitIdx), cfg);

    nuIx = find(strcmp(m.paramNames, 'nu'), 1);
    if isempty(nuIx); nu = Inf; else; nu = mean(res.theta(:, nuIx)); end

    %% -------- rolling out-of-sample predictive --------
    fp = rollingPredictiveLocal(m, r, res.theta, splitIdx, cfg);

    testIdx = (splitIdx + 1 : T)';
    rTest   = r(testIdx);
    rvProxy = max(rTest .^ 2, 1e-8);
    rvSqrt  = sqrt(rvProxy);

    [sc, ~] = scoreModelLocal(fp.varForecast, fp.logPredDensity, nu, ...
                              rTest, rvProxy, rvSqrt, cfg.alphaQS);

    %% -------- report --------
    if cfg.verbose
        fprintf('\n===== %s out-of-sample scores (lower is better) =====\n', m.name);
        fprintf('  PPS   = %+.4f\n', sc.pps);
        fprintf('  QS1   = %+.4f\n', sc.qs1);
        fprintf('  QS5   = %+.4f\n', sc.qs5);
        fprintf('  MSE   = %+.4f\n', sc.mse);
        fprintf('  MAE   = %+.4f\n', sc.mae);
        fprintf('  QLIKE = %+.4f\n', sc.qlike);
        fprintf('  logMarginalLik = %+.4f\n', res.logMarginalLik);
        pm = mean(res.theta, 1);
        for k = 1:numel(m.paramNames)
            fprintf('  posteriorMean[%-10s] = %+.4f\n', m.paramNames{k}, pm(k));
        end
    end

    out = struct('scores', sc, 'posteriorMean', mean(res.theta, 1), ...
                 'paramNames', {m.paramNames}, 'varForecast', fp.varForecast, ...
                 'nu', nu, 'logMarginalLik', res.logMarginalLik, ...
                 'meanEquation', meanInfo, 'testIdx', testIdx);
end


% =====================================================================
%                 SVM MODEL BUILDER (struct of handles)
% =====================================================================
% Every model function takes the model struct `m` as its first argument so
% the engine can call m.fn(m, ...) without closures. theta layout matches the
% tested +models classes exactly.

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


% ---- shared latent init (stationary) ----
function h0 = initLatentCommon(~, theta, n)
    mu = theta(1); phi = theta(2); se = theta(3);
    h0 = mu + se / sqrt(1 - phi^2) * randn(n, 1);
end

% ---- observation densities ----
% obs_svm carries the variance-in-mean shift lambda*exp(h_t) (see header).
function logp = obs_svm(~, yT, hT, theta)
    meanT = theta(4) + theta(5) .* exp(hT);
    logp  = -0.5*log(2*pi) - 0.5*hT - 0.5*(yT - meanT).^2 .* exp(-hT);
end

% ---- transition ----
function [hNew, aux] = tr_SV(~, hOld, theta, ~, n, aux)
    mu = theta(1); phi = theta(2); se = theta(3);
    hNew = mu + phi*(hOld - mu) + se*randn(n,1);
end

% ---- auxiliary state ----
function aux = aux_none(~, ~, n);          aux = struct('n', n); end
function aux = aux_passthrough(~, aux, ~, ~, ~, ~); end

% ---- prior sampler ----
function theta = sp_SVM(~, n)
    theta = [10*randn(n,1), 2*betarnd(20,1.5,n,1)-1, abs(trnd(1,n,1)), randn(n,1), randn(n,1)];
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


% =====================================================================
%                      LEVERAGE COUPLINGS
% =====================================================================
% (Used by demoData's SVLTRECH-SRN data-generating process.)
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

% ---- minimal SRN helpers used only by the demoData generator ----
function s = unpack_srn(theta, K)
    s.mu=theta(1); s.phi=theta(2); s.sigma_eta=theta(3); s.rho=theta(4); s.nu=theta(5);
    s.beta_0=theta(6); s.beta_1=theta(7); s.v_h=theta(8); s.v_r=theta(9); s.v_omega=theta(10);
    if K>0; s.v_z=theta(11:10+K); else; s.v_z=zeros(1,0); end
    s.w_h=theta(11+K); s.b=theta(12+K);
end
function sNew = cell_srn(x, sPrev, w)
    sNew = max(x * w.v(:) + w.w_h * sPrev + w.b, 0);
end
