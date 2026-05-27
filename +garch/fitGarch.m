function fit = fitGarch(y, opts)
% fitGarch  Fit a GARCH-family benchmark via the Econometrics Toolbox.
%
%   fit = garch.fitGarch(y)
%   fit = garch.fitGarch(y, opts)
%
%   Wraps MATLAB's garch / gjr objects + estimate. Covers the §7.1-7.2
%   benchmarks (PROPOSED_METHODOLOGY.md):
%     type='garch', dist='t'    -> GARCH(1,1)-t            (§7.1)
%     type='gjr',   dist='t'    -> GJR-GARCH(1,1)-t        (§7.2)
%   Gaussian innovations available via dist='gaussian'.
%
%   GARCH-X (§7.3, exogenous regressors in the variance equation) and
%   RealGARCH (§7.4) are NOT supported by the toolbox's garch/gjr objects
%   and are handled separately (see garch.fitGarchX, deferred).
%
%   REQUIRES the Econometrics Toolbox. This is a Stage-2 (empirical
%   application) baseline run locally; it is intentionally NOT part of the
%   Colab Stage-1 bundle, whose MATLAB install ships base + Statistics only.
%
%   Inputs
%   ------
%     y    : T-by-1 mean-zero return series (subtract the mean equation first).
%     opts : struct, optional
%              .type 'garch' (default) | 'gjr'
%              .dist 't' (default) | 'gaussian'
%              .p    GARCH order (default 1)
%              .q    ARCH order   (default 1)
%
%   Output (struct `fit`)
%   ---------------------
%     .type, .dist     echoed config
%     .omega           conditional-variance constant
%     .alpha           1-by-q ARCH coefficients
%     .beta            1-by-p GARCH coefficients
%     .gamma           1-by-q leverage coefficients (gjr only; else [])
%     .nu              Student-t dof (Inf for gaussian)
%     .logLik          maximised log-likelihood
%     .aic, .bic       information criteria
%     .nParams         number of estimated parameters
%     .condVar         T-by-1 in-sample inferred conditional variances
%     .estMdl          the fitted toolbox model object (for forecast/infer)

    arguments
        y    (:,1) double
        opts struct = struct()
    end

    defaults.type = 'garch';
    defaults.dist = 't';
    defaults.p    = 1;
    defaults.q    = 1;
    opts = mergeStruct(defaults, opts);

    validatestring(opts.type, {'garch','gjr'}, mfilename, 'type');
    validatestring(opts.dist, {'t','gaussian'}, mfilename, 'dist');

    switch opts.type
        case 'garch'
            Mdl = garch(opts.p, opts.q);
        case 'gjr'
            Mdl = gjr(opts.p, opts.q);   % positional form also creates Leverage{1..q}
    end
    if strcmp(opts.dist, 't')
        Mdl.Distribution = 't';
    end

    [estMdl, ~, logL] = estimate(Mdl, y, 'Display', 'off');

    fit.type   = opts.type;
    fit.dist   = opts.dist;
    fit.omega  = estMdl.Constant;
    fit.alpha  = cell2matSafe(estMdl.ARCH);
    fit.beta   = cell2matSafe(estMdl.GARCH);
    if strcmp(opts.type, 'gjr')
        fit.gamma = cell2matSafe(estMdl.Leverage);
    else
        fit.gamma = [];
    end
    if strcmp(opts.dist, 't')
        fit.nu = estMdl.Distribution.DoF;
    else
        fit.nu = Inf;
    end

    % Estimated parameters: omega + q ARCH + p GARCH (+ q leverage) (+ 1 dof).
    k = 1 + opts.p + opts.q;
    if strcmp(opts.type, 'gjr'); k = k + opts.q; end
    if strcmp(opts.dist, 't');   k = k + 1;      end

    T = numel(y);
    fit.logLik  = logL;
    fit.nParams = k;
    fit.aic     = -2 * logL + 2 * k;
    fit.bic     = -2 * logL + k * log(T);
    fit.condVar = infer(estMdl, y);
    fit.estMdl  = estMdl;
end


function v = cell2matSafe(c)
% Toolbox stores coefficients as cell arrays; flatten to a row, [] if empty.
    if isempty(c)
        v = [];
    else
        v = reshape(cell2mat(c), 1, []);
    end
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
