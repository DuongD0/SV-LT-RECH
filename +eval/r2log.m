function score = r2log(rv, vHat)
% r2log  Hansen-Lunde (2005) R-squared LOG metric for volatility forecasts.
%
%   score = r2log(rv, vHat)
%
%   rv    : T_test-by-1 realised variance RV_t (positive).
%   vHat  : T_test-by-1 forecast standard deviation v_hat_t.
%
%   R^2_LOG = (1/T) * sum_t [ log(RV_t / vHat_t^2) ]^2
%
%   Lower is better. Penalises proportional rather than absolute error.

    arguments
        rv   (:,1) double {mustBePositive}
        vHat (:,1) double {mustBePositive}
    end
    score = mean(log(rv ./ vHat.^2).^2);
end
