function out = residualDiagnostics(e, lags, archLags)
% residualDiagnostics  Battery of residual tests (LB, McLeod-Li, ARCH-LM, JB).
%
%   out = residualDiagnostics(e)
%   out = residualDiagnostics(e, lags, archLags)
%
%   e        : T-by-1 residual / standardised-residual series.
%   lags     : Ljung-Box lag list, default [10, 20].
%   archLags : ARCH-LM lag list,   default [5, 10].
%
%   Returns out.<test>.h / pValue per test. Volatility-clustering evidence
%   (McLeod-Li / ARCH-LM rejection) is the prerequisite for any SV / GARCH
%   model. See PROPOSED_METHODOLOGY.md §4.

    arguments
        e        (:,1) double
        lags     (1,:) double = [10, 20]
        archLags (1,:) double = [5, 10]
    end

    %% Ljung-Box on residuals — H0: no autocorrelation
    [out.lb.h, out.lb.pValue] = lbqtest(e, 'Lags', lags);

    %% McLeod-Li (Ljung-Box on squared residuals) — H0: no ARCH effect
    [out.mcleodli.h, out.mcleodli.pValue] = lbqtest(e.^2, 'Lags', lags);

    %% Engle's ARCH-LM — H0: homoscedasticity
    [out.archlm.h, out.archlm.pValue] = archtest(e, 'Lags', archLags);

    %% Jarque-Bera — H0: normality
    [out.jb.h, out.jb.pValue] = jbtest(e);

    %% Higher moments (for paper Table T1)
    out.moments.mean     = mean(e);
    out.moments.std      = std(e);
    out.moments.skewness = skewness(e);
    out.moments.kurtosis = kurtosis(e);
end
