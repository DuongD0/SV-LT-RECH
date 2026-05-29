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
