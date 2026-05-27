function out = meanEquation(y, opts)
% meanEquation  μ=0 default, ARMA(p,q) fallback if Ljung-Box rejects.
%
%   out = meanEquation(y)
%   out = meanEquation(y, opts)
%
%   Inputs
%   ------
%     y     : T-by-1 return series.
%     opts  : struct
%               .alpha     Ljung-Box significance level (default 0.05)
%               .ljungLag  LB lag used for the gate (default 10)
%               .maxP      ARMA AR max order to search (default 3)
%               .maxQ      ARMA MA max order to search (default 3)
%               .force     'zero' | 'arma' | 'auto' (default 'auto')
%
%   Outputs
%   -------
%     out.residuals  : T-by-1 mean-equation residuals epsilon_hat
%     out.usedARMA   : logical
%     out.p, out.q   : chosen orders (0 if usedARMA == false)
%     out.bic        : selected BIC (NaN if zero-mean)
%     out.ljungH     : Ljung-Box decision on RAW returns at ljungLag
%     out.ljungP     : Ljung-Box p-value on RAW returns at ljungLag
%
%   Protocol (PROPOSED_METHODOLOGY.md §3):
%     1. Run Ljung-Box on y at lag = ljungLag.
%     2. If LB fails to reject (white noise) AND opts.force ~= 'arma',
%        use mu = 0 -> residuals = y.
%     3. Else fit ARMA(p,q) over p,q in {0..maxP} x {0..maxQ}, pick by
%        BIC, and return residuals from the fit.

    arguments
        y     (:,1) double
        opts        struct = struct()
    end

    defaults.alpha    = 0.05;
    defaults.ljungLag = 10;
    defaults.maxP     = 3;
    defaults.maxQ     = 3;
    defaults.force    = 'auto';
    opts = mergeStruct(defaults, opts);

    %% Ljung-Box gate on RAW returns. h = 1 means reject H0 (autocorr present).
    [ljungH, ljungP] = lbqtest(y, 'Lags', opts.ljungLag, 'Alpha', opts.alpha);

    useArma = ljungH(end) && ~strcmp(opts.force, 'zero');
    if strcmp(opts.force, 'arma'); useArma = true; end

    if ~useArma
        out.residuals = y;
        out.usedARMA  = false;
        out.p         = 0;
        out.q         = 0;
        out.bic       = NaN;
        out.ljungH    = ljungH(end);
        out.ljungP    = ljungP(end);
        return
    end

    %% BIC search over (p,q)
    bestBic   = inf;
    bestModel = [];
    bestPQ    = [0, 0];
    for p = 0:opts.maxP
        for q = 0:opts.maxQ
            if p == 0 && q == 0; continue; end
            try
                mdl = arima('ARLags', 1:p, 'MALags', 1:q, 'Constant', NaN);
                [est, ~, logL] = estimate(mdl, y, 'Display', 'off');
                nParams = p + q + 2;                 % AR + MA + constant + variance
                bic     = -2 * logL + nParams * log(numel(y));
                if bic < bestBic
                    bestBic   = bic;
                    bestModel = est;
                    bestPQ    = [p, q];
                end
            catch
                continue
            end
        end
    end

    if isempty(bestModel)
        warning('meanEquation:noFit', ...
            'No ARMA model converged; falling back to zero-mean.');
        out.residuals = y;
        out.usedARMA  = false;
        out.p = 0; out.q = 0;
        out.bic = NaN;
        out.ljungH = ljungH(end);
        out.ljungP = ljungP(end);
        return
    end

    out.residuals = infer(bestModel, y);
    out.usedARMA  = true;
    out.p         = bestPQ(1);
    out.q         = bestPQ(2);
    out.bic       = bestBic;
    out.ljungH    = ljungH(end);
    out.ljungP    = ljungP(end);
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
