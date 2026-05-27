function loss = qlike(sigma2Hat, sigma2Actual)
% qlike  Quasi-likelihood loss for volatility forecast comparison.
%
%   loss = qlike(sigma2Hat, sigma2Actual)
%
%   QLIKE(sigma2Hat, sigma2Actual) = log(sigma2Hat) + sigma2Actual / sigma2Hat
%
%   Robust to noise in the volatility proxy (Patton 2011). Asymmetric:
%   penalises under-prediction (small sigma2Hat) more than over-prediction.
%   The PRIMARY loss for DM tests and MCS in this codebase.
%
%   Returns mean loss over t when inputs are vectors.

    arguments
        sigma2Hat    (:,1) double {mustBePositive}
        sigma2Actual (:,1) double {mustBeNonnegative}
    end
    loss = mean(log(sigma2Hat) + sigma2Actual ./ sigma2Hat);
end
