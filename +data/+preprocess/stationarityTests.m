function out = stationarityTests(y, alpha)
% stationarityTests  Run ADF, PP, KPSS on a series and combine the verdict.
%
%   out = stationarityTests(y)
%   out = stationarityTests(y, alpha)
%
%   y     : T-by-1 series.
%   alpha : significance level for each test (default 0.05).
%
%   Returns a struct:
%     out.adf.h, out.adf.pValue, out.adf.lags
%     out.pp .h, out.pp .pValue
%     out.kpss.h, out.kpss.pValue
%     out.stationary  : true when ADF + PP both reject AND KPSS fails to reject
%     out.conflict    : true when verdicts disagree
%
%   Uses adftest / pptest / kpsstest from the Econometrics Toolbox.
%   See PROPOSED_METHODOLOGY.md §2.

    arguments
        y     (:,1) double
        alpha (1,1) double {mustBeInRange(alpha, 0, 1)} = 0.05
    end

    %% ADF — H0: unit root
    [out.adf.h, out.adf.pValue] = adftest(y, 'Alpha', alpha);

    %% Phillips-Perron — H0: unit root
    [out.pp.h, out.pp.pValue] = pptest(y, 'Alpha', alpha);

    %% KPSS — H0: stationarity
    [out.kpss.h, out.kpss.pValue] = kpsstest(y, 'Alpha', alpha);

    %% Combined verdict (PROPOSED_METHODOLOGY.md §2.2)
    out.stationary = out.adf.h && out.pp.h && ~out.kpss.h;
    out.conflict   = (out.adf.h ~= out.pp.h) || (out.adf.h == out.kpss.h);
end
