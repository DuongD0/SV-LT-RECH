function T = tableT7residuals(bundle)
% tableT7residuals  T7: standardized-residual moments + Ljung-Box on squared
%   residuals (leftover ARCH) per model. Reads bundle.models(m).stdResid.
    arguments
        bundle (1,1) struct
    end
    M = bundle.models;
    name = {}; mu = []; sd = []; sk = []; ku = []; lbq2 = [];
    for m = 1:numel(M)
        e = M(m).stdResid(:);
        e = e(isfinite(e));
        rd = data.preprocess.residualDiagnostics(e);
        name{end+1,1} = M(m).name;            %#ok<AGROW>
        mu(end+1,1)   = rd.moments.mean;       %#ok<AGROW>
        sd(end+1,1)   = rd.moments.std;        %#ok<AGROW>
        sk(end+1,1)   = rd.moments.skewness;   %#ok<AGROW>
        ku(end+1,1)   = rd.moments.kurtosis;   %#ok<AGROW>
        p = rd.mcleodli.pValue;                % Ljung-Box on squared resids
        lbq2(end+1,1) = p(1);                  %#ok<AGROW>
    end
    T = table(name, mu, sd, sk, ku, lbq2, ...
        'VariableNames', {'Model','Mean','Std','Skewness','Kurtosis','LBQ2_pValue'});
end
