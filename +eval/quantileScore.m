function score = quantileScore(y, qHat, alpha)
% quantileScore  Quantile score for VaR-style forecast evaluation.
%
%   score = quantileScore(y, qHat, alpha)
%
%   y     : T-by-1 realised returns.
%   qHat  : T-by-1 alpha-quantile forecasts (e.g. VaR at level alpha).
%   alpha : scalar quantile level in (0, 1).
%
%   QS = (1/T) * sum_t (alpha - 1{y_t <= q_t}) * (y_t - q_t)
%
%   Lower is better. See PROPOSED_METHODOLOGY.md §10.1 (Taylor 2019).

    arguments
        y     (:,1) double
        qHat  (:,1) double
        alpha (1,1) double {mustBeInRange(alpha, 0, 1)}
    end
    indicator = double(y <= qHat);
    score = mean((alpha - indicator) .* (y - qHat));
end
