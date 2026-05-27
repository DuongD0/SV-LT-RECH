function out = dieboldMariano(loss1, loss2, hForecast)
% dieboldMariano  Diebold-Mariano test of equal predictive accuracy.
%
%   out = dieboldMariano(loss1, loss2, hForecast)
%
%   loss1, loss2 : T_test-by-1 loss series from model 1 and model 2.
%   hForecast    : forecast horizon (default 1 for one-step-ahead).
%
%   Test:
%     d_t      = loss1_t - loss2_t
%     dbar     = mean(d_t)
%     var_dbar = Newey-West HAC variance with bandwidth floor(T^(1/3))
%     DM       = dbar / sqrt(var_dbar / T)   ~ N(0,1) under H0
%
%   Returns out.statistic, out.pValue (two-sided), out.meanDiff,
%   out.bandwidth.
%
%   See Diebold & Mariano (1995); HAC variance from Newey-West (1987).
%   PROPOSED_METHODOLOGY.md §10.2.

    arguments
        loss1     (:,1) double
        loss2     (:,1) double
        hForecast (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    assert(numel(loss1) == numel(loss2), 'dieboldMariano:lengthMismatch');

    T         = numel(loss1);
    d         = loss1 - loss2;
    dbar      = mean(d);
    dCentered = d - dbar;

    %% Newey-West HAC variance estimate
    bandwidth = max(hForecast - 1, floor(T^(1/3)));
    gamma0    = mean(dCentered .^ 2);
    s         = gamma0;
    for k = 1:bandwidth
        w     = 1 - k / (bandwidth + 1);                 % Bartlett kernel
        gamma = mean(dCentered(1:end-k) .* dCentered(k+1:end));
        s     = s + 2 * w * gamma;
    end
    varDbar = s / T;

    out.statistic = dbar / sqrt(max(varDbar, 1e-12));
    out.pValue    = 2 * (1 - normcdf(abs(out.statistic)));
    out.meanDiff  = dbar;
    out.bandwidth = bandwidth;
end
