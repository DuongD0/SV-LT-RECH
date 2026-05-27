function score = pps(logPredictiveDensity)
% pps  Partial Predictive Score (negative mean log predictive density).
%
%   score = pps(logPredictiveDensity)
%
%   logPredictiveDensity : T_test-by-1 vector of log p(y_t | y_{1:t-1}).
%   score                : -mean(logPredictiveDensity). Lower is better.
%
%   See PROPOSED_METHODOLOGY.md §10.1.

    arguments
        logPredictiveDensity (:,1) double
    end
    score = -mean(logPredictiveDensity);
end
