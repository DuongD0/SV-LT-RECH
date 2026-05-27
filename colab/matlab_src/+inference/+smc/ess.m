function [essVal, logEss] = ess(logW)
% ess  Effective sample size from log-weights.
%
%   essVal           = ess(logW)
%   [essVal, logEss] = ess(logW)
%
%   ESS = 1 / sum(w_i^2)  with w_i = exp(logW_i) / sum_j exp(logW_j)
%
%   In log-space:
%       log ESS = 2 logsumexp(logW) - logsumexp(2 logW)
%
%   This works whether logW is pre-normalised or not.

    logZ   = utils.logsumexp(logW(:));
    logSq  = utils.logsumexp(2 * logW(:));
    logEss = 2 * logZ - logSq;
    essVal = exp(logEss);
end
