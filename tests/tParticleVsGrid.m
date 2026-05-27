classdef tParticleVsGrid < matlab.unittest.TestCase
% tParticleVsGrid  Bootstrap PF log-likelihood vs brute-force grid filter.
%
%   At fixed theta on a short synthetic SV series, the particle filter
%   marginal log-likelihood should agree with a 1,000-point Riemann-sum
%   grid filter on h in [-10, 10] to within ~0.5 nats.
%
%   This is the cheapest test that catches the worst class of bugs
%   (log-weight sign flips, missing logsumexp, off-by-one in resampling,
%   un-normalised proposal density).

    methods (Test)

        function plainSvAgreesWithGrid(testCase)
            seed = 20260516;
            utils.reproducibility(seed);

            mdl   = models.SV();
            theta = [0.0, 0.95, 0.4];
            T     = 100;

            [y, ~] = mdl.simulate(theta, T);

            %% Average bootstrap PF over multiple runs for low MC variance
            nRuns  = 20;
            pfOpts = struct('M', 1000, 'essThreshold', 0.5);
            pfLogLiks = zeros(nRuns, 1);
            for k = 1:nRuns
                pfLogLiks(k) = inference.pf.bootstrap(mdl, y, theta, pfOpts);
            end
            pfMean = mean(pfLogLiks);

            %% Brute-force grid filter
            gridLogLik = gridFilter(mdl, y, theta, -10, 10, 1000);

            testCase.verifyEqual(pfMean, gridLogLik, 'AbsTol', 0.5, ...
                sprintf('PF mean %.3f vs grid %.3f', pfMean, gridLogLik));
        end
    end
end


function logLik = gridFilter(mdl, y, theta, hLo, hHi, nGrid)
% Brute-force Riemann-sum filter on a uniform h grid. Slow but correct.
    h     = linspace(hLo, hHi, nGrid)';
    dh    = h(2) - h(1);
    T     = numel(y);
    s     = mdl.unpack(theta);

    sd1   = s.sigma_eta / sqrt(1 - s.phi^2);
    logP  = -0.5*log(2*pi*sd1^2) - 0.5*((h - s.mu)/sd1).^2;
    logP  = logP - utils.logsumexp(logP + log(dh));

    logLik = 0;
    for t = 1:T
        logObs       = mdl.observationLogLik(y(t), h, theta);
        logIntegrand = logP + logObs;
        logZ_t       = utils.logsumexp(logIntegrand) + log(dh);
        logLik       = logLik + logZ_t;
        logP         = logIntegrand - logZ_t;

        if t < T
            transMean = s.mu + s.phi * (h - s.mu);
            diff      = (h' - transMean) / s.sigma_eta;
            logTrans  = -0.5*log(2*pi*s.sigma_eta^2) - 0.5*diff.^2;
            logP      = utils.logsumexp(logTrans + logP, 1)' + log(dh);
            logP      = logP - utils.logsumexp(logP + log(dh));
        end
    end
end
