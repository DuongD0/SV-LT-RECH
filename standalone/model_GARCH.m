function out = model_GARCH(y, splitIdx, type, opts)
% MODEL_GARCH  Classic GARCH(1,1)-t / GJR(1,1)-t volatility baseline.
% =====================================================================
% SELF-CONTAINED, NO DEPENDENCIES. This file fits a *classic* (no deep
% learning) GARCH-family baseline by maximum likelihood, produces an
% out-of-sample one-step-ahead conditional-variance forecast, and scores
% it. It is a verbatim extraction of the GARCH-baseline path from the
% reviewed run_comparison.m; all local functions are copied unchanged and
% live in THIS file, so there is no dependency on run_comparison.m or any
% +package.
%
% USAGE
% -----
%   model_GARCH                              % built-in synthetic demo, GARCH-t
%   out = model_GARCH(y)                     % your T-by-1 returns; 80/20 split
%   out = model_GARCH(y, splitIdx)
%   out = model_GARCH(y, splitIdx, type)     % type: 'garch' (default) | 'gjr'
%   out = model_GARCH(y, splitIdx, type, opts)
%
%   type : 'garch' for symmetric GARCH(1,1)-t, 'gjr' for GJR(1,1)-t.
%
% RETURNS  out struct with fields:
%   .type .scores (struct: pps,qs1,qs5,mse,mae,r2log,qlike)
%   .varForecast .nu .meanEquation .testIdx
%
% REQUIREMENTS
%   Econometrics Toolbox (garch/gjr/estimate/infer) for the GARCH fit and
%   for the optional ARMA mean-equation gate; Statistics & ML Toolbox for
%   the t-distribution quantities. If the Econometrics Toolbox is absent
%   the mean equation falls back to zero-mean, but the GARCH fit itself
%   requires it.
% =====================================================================

    %% -------- argument handling + demo data --------
    if nargin < 4 || isempty(opts); opts = struct(); end %#ok<NASGU>
    if nargin < 3 || isempty(type); type = 'garch'; end
    if nargin < 1 || isempty(y)
        [y, ~, splitIdx] = demoData();   % demoData returns [y,Z,splitIdx]; Z unused here
        fprintf('[model_GARCH] No data supplied -> built-in synthetic demo.\n');
    else
        if nargin < 2 || isempty(splitIdx); splitIdx = floor(0.8 * numel(y)); end
    end

    y = y(:);
    T = numel(y);
    if nargin < 2 || isempty(splitIdx); splitIdx = floor(0.8 * numel(y)); end
    assert(splitIdx > 10 && splitIdx < T, 'splitIdx out of range');

    hasEcon = ~isempty(ver('econ'));

    %% -------- mean equation (ARMA auto-gate) on residuals --------
    [r, meanInfo] = meanEquationLocal(y, splitIdx, 'auto', hasEcon);

    testIdx = (splitIdx + 1 : T)';
    rTest   = r(testIdx);
    rvProxy = max(rTest .^ 2, 1e-8);
    rvSqrt  = sqrt(rvProxy);
    alphaQS = [0.01 0.05];

    %% -------- fit + roll forecast --------
    rTrain = r(1:splitIdx);
    fit = fitGarchLocal(rTrain, type);
    rf  = rollingForecastGarchLocal(fit, r, splitIdx);
    nu  = fit.nu;
    varForecast = rf.sigma2Forecast;
    logPredDensity = rf.logPredDensity;

    %% -------- score --------
    [sc, ~] = scoreModelLocal(varForecast, logPredDensity, nu, ...
                              rTest, rvProxy, rvSqrt, alphaQS);

    fprintf('[model_GARCH:%s] PPS=%.4f  QLIKE=%.4f  MSE=%.6f  MAE=%.6f\n', ...
        type, sc.pps, sc.qlike, sc.mse, sc.mae);

    out = struct('type', type, 'scores', sc, 'varForecast', varForecast, ...
                 'nu', nu, 'meanEquation', meanInfo, 'testIdx', testIdx);
end


% =====================================================================
%                      GARCH FAMILY (classic)
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
%                            DEMO DATA
% =====================================================================
% demoData() and its transitive helpers (build_RECH, rechNames, unpack_srn,
% cell_srn, stableSigmoid, leverageCouple, ocsnComponents, and the handle
% targets referenced by build_RECH) are copied VERBATIM from
% run_comparison.m so the synthetic demo is fully self-contained. demoData
% returns [y, Z, splitIdx]; this file ignores Z.
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


% ---- demoData support: SV/RECH builder + unpackers + RNN cell + leverage --
% (verbatim from run_comparison.m; needed so demoData is self-contained)
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

function s = unpack_srn(theta, K)
    s.mu=theta(1); s.phi=theta(2); s.sigma_eta=theta(3); s.rho=theta(4); s.nu=theta(5);
    s.beta_0=theta(6); s.beta_1=theta(7); s.v_h=theta(8); s.v_r=theta(9); s.v_omega=theta(10);
    if K>0; s.v_z=theta(11:10+K); else; s.v_z=zeros(1,0); end
    s.w_h=theta(11+K); s.b=theta(12+K);
end

function sNew = cell_srn(x, sPrev, w)
    sNew = max(x * w.v(:) + w.w_h * sPrev + w.b, 0);
end

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


% ---- handle targets referenced (not called) by build_RECH; copied verbatim
%      so the @-handles in build_RECH resolve. -----------------------------
function h0 = initLatentCommon(~, theta, n)
    mu = theta(1); phi = theta(2); se = theta(3);
    h0 = mu + se / sqrt(1 - phi^2) * randn(n, 1);
end

function logp = obs_scaledt(~, yT, hT, theta)
    nu = theta(5);
    scaledSq = yT.^2 ./ (exp(hT) * (nu - 2));
    logp = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(pi*(nu-2)) ...
         - 0.5*hT - ((nu+1)/2)*log1p(scaledSq);
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
function lp = lp_SVLT(~, theta)
    if theta(4)<=-1 || theta(4)>=1; lp=-Inf; return; end
    lp = lp_SVt_core([theta(1:3), theta(5)]) - log(2);
end
function lp = lp_SVt_core(theta)   % [mu phi se nu]
    nu = theta(4);
    if theta(2)<=-1||theta(2)>=1||theta(3)<=0||nu<=2; lp=-Inf; return; end
    lp = lp_SV([], theta(1:3)) + log(0.1) - 0.1*nu + 0.1*2;   % Exp(0.1) trunc nu>2
end
function lp = lp_SV(~, theta)
    mu=theta(1); phi=theta(2); se=theta(3);
    if phi<=-1||phi>=1||se<=0; lp=-Inf; return; end
    lp = -0.5*(mu/10)^2 - log(10) - 0.5*log(2*pi);
    ph=(1+phi)/2; lp = lp + (20-1)*log(ph) + (1.5-1)*log(1-ph) - betaln(20,1.5) - log(2);
    lp = lp + log(2) - log(pi) - log(1+se^2);
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

function out = ifelse(cond, a, b); if cond; out = a; else; out = b; end; end
