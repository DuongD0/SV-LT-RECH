function [thetaNew, logLikNew, accept] = rwMetropolis(model, theta, logLik, y, aK, proposalCov, pfOpts, nSweeps)
% rwMetropolis  Random-walk Metropolis rejuvenation for SMC parameter particles.
%
%   [thetaNew, logLikNew, accept] = rwMetropolis(model, theta, logLik, ...
%                                               y, aK, proposalCov, pfOpts, nSweeps)
%
%   Performs nSweeps Metropolis updates of each parameter particle from
%   the tempered posterior
%
%       pi_k(theta) propto p(theta) * p(y|theta)^aK
%
%   logLik for each particle is recomputed via the bootstrap PF when a
%   proposal is accepted.
%
%   Inputs
%   ------
%     model       : models.Model handle.
%     theta       : N-by-K matrix of parameter particles.
%     logLik      : N-by-1 log-marginal-likelihoods at each theta_i.
%     y           : T-by-1 return series.
%     aK          : current anneal temperature.
%     proposalCov : K-by-K RWM proposal covariance.
%     pfOpts      : struct passed through to inference.pf.bootstrap.
%     nSweeps     : number of Metropolis sweeps (default 10).
%
%   Outputs
%   -------
%     thetaNew  : N-by-K updated particles.
%     logLikNew : N-by-1 updated log-likelihoods.
%     accept    : scalar fraction of accepted proposals across all sweeps.

    arguments
        model               models.Model
        theta       (:,:)   double
        logLik      (:,1)   double
        y           (:,1)   double
        aK          (1,1)   double
        proposalCov (:,:)   double
        pfOpts              struct = struct()
        nSweeps     (1,1)   double {mustBePositive} = 10
    end

    [N, K]    = size(theta);
    L         = chol(proposalCov, 'lower');
    thetaNew  = theta;
    logLikNew = logLik;

    nAccept = 0;
    nTotal  = 0;

    for sweep = 1:nSweeps
        for i = 1:N
            thetaCurr = thetaNew(i, :);
            llhCurr   = logLikNew(i);
            lpCurr    = model.logPrior(thetaCurr);

            thetaProp = (thetaCurr' + L * randn(K, 1))';
            lpProp    = model.logPrior(thetaProp);

            if isinf(lpProp)
                nTotal = nTotal + 1;
                continue
            end

            llhProp = inference.pf.bootstrap(model, y, thetaProp, pfOpts);
            logR    = (lpProp + aK * llhProp) - (lpCurr + aK * llhCurr);

            if log(rand()) < logR
                thetaNew(i, :) = thetaProp;
                logLikNew(i)   = llhProp;
                nAccept        = nAccept + 1;
            end
            nTotal = nTotal + 1;
        end
    end

    accept = nAccept / max(nTotal, 1);
end
