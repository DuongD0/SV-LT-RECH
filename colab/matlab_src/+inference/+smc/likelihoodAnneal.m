function result = likelihoodAnneal(model, y, opts)
% likelihoodAnneal  Likelihood-annealing SMC for Bayesian posterior + log-Z.
%
%   result = likelihoodAnneal(model, y, opts)
%
%   Samples from the posterior pi_K(theta) = p(theta|y) via a sequence
%   of tempered targets
%
%       pi_k(theta) propto p(theta) * p(y|theta)^{a_k},
%       0 = a_1 < a_2 < ... < a_K = 1.
%
%   Each step:
%     1. Reweight by L(theta_i)^{a_k - a_{k-1}}.
%     2. If conditional ESS drops below threshold, resample particles.
%     3. Apply nSweeps random-walk Metropolis sweeps at temperature a_k.
%
%   The annealing schedule is adaptive: pick a_k so that conditional ESS
%   equals targetEss (default N/2). See Duan & Fulop (2015), and
%   PROPOSED_METHODOLOGY.md §9.2.
%
%   Inputs
%   ------
%     model : models.Model handle.
%     y     : T-by-1 return series.
%     opts  : struct
%               .N             parameter-particle count (default 5000)
%               .M             latent-particle count for PF (default 200)
%               .essThreshold  fraction of N for resample trigger (default 0.5)
%               .targetEss     temperature-bisection target (default 0.5)
%               .nSweeps       RWM sweeps per anneal step (default 10)
%               .proposalScale scale on empirical cov (default 0.5)
%               .pfOpts        passed straight to inference.pf.bootstrap
%               .verbose       print progress (default true)
%
%   Output struct fields
%   --------------------
%     .theta            : N-by-K final posterior particles
%     .logLik           : N-by-1 log p(y|theta_i)
%     .logMarginalLik   : scalar log p(y)
%     .schedule         : K-by-1 anneal temperatures a_1..a_K
%     .essTrace         : ESS at every step
%     .acceptTrace      : RWM acceptance fraction per step
%     .seedRNGState     : RNG state at entry

    arguments
        model           models.Model
        y       (:,1)   double
        opts            struct = struct()
    end

    defaults.N             = 5000;
    defaults.M             = 200;
    defaults.essThreshold  = 0.5;
    defaults.targetEss     = 0.5;
    defaults.nSweeps       = 10;
    defaults.proposalScale = 0.5;
    defaults.pfOpts        = struct();
    defaults.verbose       = true;
    opts = mergeStruct(defaults, opts);

    N = opts.N;
    K = model.nParams();
    pfOpts = mergeStruct(struct('M', opts.M), opts.pfOpts);

    %% Initialise from prior + score every particle via PF
    rngStateAtEntry = rng();
    theta = model.samplePrior(N);

    logLik = zeros(N, 1);
    for i = 1:N
        logLik(i) = inference.pf.bootstrap(model, y, theta(i, :), pfOpts);
    end

    logW        = zeros(N, 1) - log(N);
    a           = 0;
    schedule    = 0;
    essTrace    = N;
    acceptTrace = [];
    logZ        = 0;

    step = 0;
    while a < 1
        step = step + 1;

        %% Find next temperature
        [da, ~] = inference.smc.adaptiveTemperature(...
            logW, logLik, 1 - a, opts.targetEss * N);
        aNew = a + da;

        %% Reweight + accumulate marginal-likelihood contribution
        logWincrement = da * logLik;
        logZstep      = utils.logsumexp(logW + logWincrement);
        logZ          = logZ + logZstep;
        logW          = logW + logWincrement - logZstep;

        %% Resample if ESS too low
        currentEss      = inference.smc.ess(logW);
        essTrace(end+1) = currentEss; %#ok<AGROW>
        if currentEss < opts.essThreshold * N
            idx    = inference.pf.resampleSystematic(logW);
            theta  = theta(idx, :);
            logLik = logLik(idx);
            logW   = zeros(N, 1) - log(N);
        end

        %% Adaptive RWM proposal covariance
        proposalCov = opts.proposalScale^2 * cov(theta) + 1e-8 * eye(K);

        %% Markov rejuvenation
        [theta, logLik, acceptFrac] = inference.mcmc.rwMetropolis(...
            model, theta, logLik, y, aNew, proposalCov, pfOpts, opts.nSweeps);
        acceptTrace(end+1) = acceptFrac; %#ok<AGROW>

        if opts.verbose
            fprintf('  step %3d  a=%.4f  ESS=%.0f  accept=%.2f\n', ...
                step, aNew, currentEss, acceptFrac);
        end

        a = aNew;
        schedule(end+1) = a; %#ok<AGROW>
    end

    result.theta          = theta;
    result.logLik         = logLik;
    result.logMarginalLik = logZ;
    result.schedule       = schedule(:);
    result.essTrace       = essTrace(:);
    result.acceptTrace    = acceptTrace(:);
    result.seedRNGState   = rngStateAtEntry;
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
