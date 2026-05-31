function fit = fitGarchRECH(y, opts)
% fitGarchRECH  GARCH-RECH (Recurrent Conditional Heteroskedasticity) by MLE.
%
%   fit = garch.fitGarchRECH(y)
%   fit = garch.fitGarchRECH(y, opts)
%
%   The "GARCH Deep-Learning" baseline: the original RECH model of
%   Nguyen, Tran & Kohn (2022, J. Appl. Econometrics 37(5)). It is the
%   GARCH-backbone analogue of this codebase's SV-backbone SVLTRECH, so the
%   two give a clean {GARCH, SV} x {no-DL, DL} effectiveness comparison.
%
%   Model (RECH(1,1), §7.6)
%   -----------------------
%     r_t       = sigma_t * eps_t,        eps_t ~ scaled t_nu  (or N(0,1))
%     sigma2_t  = omega_t + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
%     omega_t   = beta_0 + beta_1 * s_t                 (RNN-driven "constant")
%     s_t       = RNNcell( x_t, s_{t-1} ),  s_1 == 0
%     x_t       = [ sigma2_{t-1}, r_{t-1}, omega_{t-1}, z_{t-1} ]
%
%   The recurrent cell is one of the shared +cells primitives:
%     'srn'  -> cells.srn   (RECH-SRN, Nguyen-Tran-Kohn 2022 Eq. 3)
%     'lstm' -> cells.lstm  (RECH-LSTM)
%     'gru'  -> cells.gru   (RECH-GRU)
%
%   sigma2_t is a DETERMINISTIC function of (theta, data) -- there is no
%   latent state to integrate, so the conditional likelihood is exact and
%   the model is fit by numerical maximum likelihood (unlike the SV family,
%   which needs SMC). Optimisation runs on an unconstrained reparameterisation
%   (persistence/split logits, softplus on positives) so alpha,beta >= 0 with
%   alpha + beta < 1 and nu > 2 hold by construction; sigma2_t is floored at
%   1e-8 for numerical safety.
%
%   Inputs
%   ------
%     y    : T-by-1 mean-zero return series (subtract the mean equation first).
%     opts : struct, optional
%              .cell      'srn' (default) | 'lstm' | 'gru'
%              .dist      't'   (default) | 'gaussian'
%              .Z         T-by-K exogenous covariates in the variance eqn (default [])
%              .restarts  random multistarts for the non-convex MLE (default 5)
%              .maxEval   max optimiser function evals per start (default 4000)
%              .seedVar   sigma2_1 seed (default var(y))
%              .verbose   print per-start NLL (default false)
%
%   Output (struct `fit`)
%   ---------------------
%     .cell,.dist,.nCovariates   echoed config
%     .alpha,.beta               GARCH ARCH/GARCH coefficients
%     .beta0,.beta1              RNN "constant" intercept + loading
%     .cellWeights               struct of fitted RNN weights (cell-specific)
%     .nu                        Student-t dof (Inf for gaussian)
%     .seedVar                   sigma2_1 seed used
%     .logLik,.aic,.bic,.nParams fit quality
%     .condVar                   T-by-1 in-sample conditional-variance path
%     .omegaPath                 T-by-1 RNN-driven omega_t path (interpretable)
%
%   REQUIRES the Optimization Toolbox for fminunc; falls back to base-MATLAB
%   fminsearch automatically if it is unavailable.
%
%   See PROPOSED_METHODOLOGY.md §7.6 and +models/SVLTRECH.m (the SV analogue).

    arguments
        y    (:,1) double
        opts struct = struct()
    end

    defaults.cell     = 'srn';
    defaults.dist     = 't';
    defaults.Z        = [];
    defaults.restarts = 5;
    defaults.maxEval  = 4000;
    defaults.seedVar  = var(y);
    defaults.verbose  = false;
    opts = mergeStruct(defaults, opts);

    validatestring(opts.cell, {'srn','lstm','gru'}, mfilename, 'cell');
    validatestring(opts.dist, {'t','gaussian'},     mfilename, 'dist');

    T = numel(y);
    Z = opts.Z;
    K = size(Z, 2);
    if ~isempty(Z)
        assert(size(Z, 1) == T, 'fitGarchRECH:covShape', ...
            'Z has %d rows; expected %d', size(Z, 1), T);
    end
    d      = 3 + K;                       % RNN input dim [sigma2,r,omega,z...]
    isT    = strcmp(opts.dist, 't');
    nP     = paramCount(opts.cell, d, isT);

    objfun = @(p) negLogLik(p, y, Z, opts.cell, opts.dist, d, opts.seedVar);

    %% Multistart unconstrained MLE
    bestNll = inf;
    bestP   = [];
    for r = 1:opts.restarts
        p0 = initParams(nP, opts.cell, d, isT, r);
        [pHat, nll] = minimize(objfun, p0, opts.maxEval);
        if opts.verbose
            fprintf('  [fitGarchRECH:%s] start %d/%d  NLL=%.4f\n', ...
                opts.cell, r, opts.restarts, nll);
        end
        if isfinite(nll) && nll < bestNll
            bestNll = nll;
            bestP   = pHat;
        end
    end
    assert(~isempty(bestP), 'fitGarchRECH:noConverge', ...
        'No optimiser start produced a finite log-likelihood.');

    P = unpackParams(bestP, opts.cell, d, isT);
    [sigma2, omegaPath] = garch.rechCondVar(P, y, Z, opts.cell, opts.seedVar);

    fit.cell        = opts.cell;
    fit.dist        = opts.dist;
    fit.nCovariates = K;
    fit.alpha       = P.alpha;
    fit.beta        = P.beta;
    fit.beta0       = P.beta0;
    fit.beta1       = P.beta1;
    fit.cellWeights = P.cellWeights;
    fit.nu          = P.nu;
    fit.seedVar     = opts.seedVar;
    fit.logLik      = -bestNll;
    fit.nParams     = nP;
    fit.aic         = -2 * fit.logLik + 2 * nP;
    fit.bic         = -2 * fit.logLik + nP * log(T);
    fit.condVar     = sigma2;
    fit.omegaPath   = omegaPath;
end


% ===================================================================
% Objective: negative log-likelihood on the unconstrained parameter p
% ===================================================================
function nll = negLogLik(p, y, Z, cellType, dist, d, seedVar)
    P = unpackParams(p, cellType, d, strcmp(dist, 't'));
    sigma2 = garch.rechCondVar(P, y, Z, cellType, seedVar);
    if any(~isfinite(sigma2)) || any(sigma2 <= 0)
        nll = inf; return
    end
    if strcmp(dist, 't')
        z   = y ./ sqrt(sigma2);
        ll  = -0.5 * log(sigma2) + logStdT(z, P.nu);
    else
        ll  = -0.5 * log(2 * pi * sigma2) - 0.5 * y.^2 ./ sigma2;
    end
    nll = -sum(ll);
    if ~isfinite(nll); nll = inf; end
end


% ===================================================================
% Unconstrained <-> natural parameter map
% ===================================================================
function P = unpackParams(p, cellType, d, isT)
    % Head: persistence/split logits, beta0 (softplus), beta1 (free).
    tau  = sig(p(1));            % alpha + beta in (0,1)
    psi  = sig(p(2));            % split
    P.alpha = tau * psi;
    P.beta  = tau * (1 - psi);
    P.beta0 = softplus(p(3));
    P.beta1 = p(4);
    idx = 5;

    switch cellType
        case 'srn'
            v   = p(idx:idx+d-1);          idx = idx + d;
            w_h = p(idx);                  idx = idx + 1;
            b   = p(idx);                  idx = idx + 1;
            P.cellWeights = struct('v', v(:)', 'w_h', w_h, 'b', b);
        case 'lstm'
            gates = {'f','i','o','c'};
            W = struct();
            for g = 1:4
                gn = gates{g};
                W.(['W_' gn]) = p(idx:idx+d-1); idx = idx + d;
                W.(['u_' gn]) = p(idx);         idx = idx + 1;
                W.(['b_' gn]) = p(idx);         idx = idx + 1;
            end
            P.cellWeights = W;
        case 'gru'
            gates = {'z','r','h'};
            W = struct();
            for g = 1:3
                gn = gates{g};
                W.(['W_' gn]) = p(idx:idx+d-1); idx = idx + d;
                W.(['u_' gn]) = p(idx);         idx = idx + 1;
                W.(['b_' gn]) = p(idx);         idx = idx + 1;
            end
            P.cellWeights = W;
    end

    if isT
        P.nu = 2 + softplus(p(idx));
    else
        P.nu = Inf;
    end
end


function nP = paramCount(cellType, d, isT)
    switch cellType
        case 'srn',  nP = 4 + (d + 2);
        case 'lstm', nP = 4 + 4 * (d + 2);
        case 'gru',  nP = 4 + 3 * (d + 2);
    end
    if isT; nP = nP + 1; end
end


function p0 = initParams(nP, ~, ~, isT, restart)
    % Small random RNN weights; persistence ~0.9, modest beta0, nu ~ 6.
    p0 = 0.05 * randn(nP, 1);
    p0(1) = 2.0 + 0.3 * randn();          % sig(2) ~ 0.88 persistence
    p0(2) = -1.0 + 0.3 * randn();         % split favours beta over alpha
    p0(3) = invSoftplus(0.05 + 0.1 * rand());   % small positive beta0
    p0(4) = 0.1 * randn();                % beta1
    if isT
        p0(end) = invSoftplus(4 + 4 * rand());   % nu ~ 6-10
    end
    % Diversify multistarts beyond the first.
    if restart > 1
        p0 = p0 + 0.25 * randn(nP, 1) .* [zeros(2,1); ones(nP-2,1)];
    end
end


function [pHat, fval] = minimize(objfun, p0, maxEval)
    % Prefer fminunc (Optimization Toolbox); fall back to fminsearch (base).
    if exist('fminunc', 'file') == 2 || exist('fminunc', 'file') == 6
        try
            o = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                'Display', 'off', 'MaxFunctionEvaluations', maxEval, ...
                'MaxIterations', maxEval, 'OptimalityTolerance', 1e-6);
            [pHat, fval] = fminunc(objfun, p0, o);
            return
        catch
            % fall through to fminsearch
        end
    end
    o = optimset('Display', 'off', 'MaxFunEvals', maxEval, 'MaxIter', maxEval);
    [pHat, fval] = fminsearch(objfun, p0, o);
end


% ===================================================================
% Numeric helpers
% ===================================================================
function y = sig(z)
    y = 0.5 * (1 + tanh(z / 2));               % stable logistic
end

function y = softplus(z)
    y = log1p(exp(-abs(z))) + max(z, 0);       % stable log(1+e^z)
end

function z = invSoftplus(y)
    z = log(expm1(max(y, 1e-8)));              % inverse of softplus
end

function lp = logStdT(z, nu)
% Log density of a unit-variance standardised Student-t (matches the SV
% models' observation density): z*sqrt(nu/(nu-2)) ~ t_nu.
    lp = gammaln((nu + 1) / 2) - gammaln(nu / 2) ...
       - 0.5 * log((nu - 2) * pi) ...
       - ((nu + 1) / 2) * log1p(z.^2 / (nu - 2));
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
