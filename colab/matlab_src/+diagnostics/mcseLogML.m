function [mcse, mean_logZ] = mcseLogML(logZChains)
% mcseLogML  Monte-Carlo standard error of log marginal likelihood.
%
%   [mcse, mean_logZ] = mcseLogML(logZChains)
%
%   logZChains : C-by-1 vector of log marginal likelihoods, one per
%                independent SMC run (use `result.logMarginalLik`).
%
%   The §4.5 gate requires MCSE < 0.5 nats across 3 seeds. Returns
%   NaN MCSE when fewer than 2 chains are provided.

    arguments
        logZChains (:,1) double
    end

    C         = numel(logZChains);
    mean_logZ = mean(logZChains);
    if C < 2
        mcse = NaN;
        return
    end
    mcse = std(logZChains) / sqrt(C);
end
