function [logLik, hFiltered] = bootstrap(model, y, theta, opts)
% bootstrap  Bootstrap particle filter for SV-family latent log-variance.
%
%   logLik              = bootstrap(model, y, theta)
%   [logLik, hFiltered] = bootstrap(model, y, theta, opts)
%
%   Inputs
%   ------
%     model   : a models.Model handle implementing initLatent,
%               transitionSample, observationLogLik.
%     y       : T-by-1 return series.
%     theta   : 1-by-K parameter row vector.
%     opts    : struct with optional fields
%                 .M             number of particles (default 200)
%                 .essThreshold  resample when ESS < essThreshold*M
%                                (default 0.5)
%                 .resample      'systematic' | 'stratified'
%                                (default 'systematic')
%                 .clipLogWeight clamp per-step log-weights below this
%                                (default -Inf -> no clip)
%                 .returnPath    true -> also return filtered states
%                                (default false)
%
%   Outputs
%   -------
%     logLik    : scalar log p(y_{1:T} | theta).
%     hFiltered : T-by-1 posterior-mean filtered latent state (if requested).
%
%   Numerical notes
%   ---------------
%   - All weights live in log-space; we never store raw weights.
%   - ESS is computed from normalised log-weights via logsumexp.
%   - Underflow clipping is OFF by default for plain SV (Gaussian
%     innovations); enabled by the Student-t / leverage / RNN models.
%
%   See PROPOSED_METHODOLOGY.md §9.3.

    arguments
        model               models.Model
        y           (:,1)   double
        theta       (1,:)   double
        opts                struct = struct()
    end

    %% Resolve options
    defaults.M             = 200;
    defaults.essThreshold  = 0.5;
    defaults.resample      = 'systematic';
    defaults.clipLogWeight = -Inf;
    defaults.returnPath    = false;
    opts = mergeStruct(defaults, opts);

    M = opts.M;
    T = numel(y);

    %% Initialise particles from the model's stationary distribution
    h    = model.initLatent(theta, M);
    aux  = model.initParticleAux(theta, M);
    logW = -log(M) * ones(M, 1);
    logLik = 0;

    if opts.returnPath
        hFiltered = zeros(T, 1);
    else
        hFiltered = [];
    end

    switch opts.resample
        case 'systematic'
            resampler = @inference.pf.resampleSystematic;
        case 'stratified'
            resampler = @inference.pf.resampleStratified;
        otherwise
            error('bootstrap:badResample', 'Unknown resampler %s', opts.resample);
    end

    %% Main filter loop
    for t = 1:T

        % Propagate: h_t | h_{t-1} (returns possibly-updated aux when
        % the model needs to persist intermediates such as the RNN
        % hidden state).
        if t > 1
            [h, aux] = model.transitionSample(h, theta, t, M, aux);
        end

        % Observation increment: log p(y_t | h_t, theta)
        logW_inc = model.observationLogLik(y(t), h, theta);
        if opts.clipLogWeight > -Inf
            logW_inc = max(logW_inc, opts.clipLogWeight);
        end

        % Refresh per-particle auxiliary state (e.g. epsilonPrev for
        % leverage, sPrev/omegaPrev for the RNN cell). Default models
        % return aux unchanged.
        aux = model.updateParticleAux(aux, h, y(t), theta, t);

        % Accumulate predictive log-likelihood:
        %   log p(y_t | y_{1:t-1}) = log( sum_i w_{t-1}^{(i)} * p(y_t|h_t^{(i)}) )
        % Using logsumexp over (current normalised logW) + observation increment.
        logLikIncrement = utils.logsumexp(logW + logW_inc);
        logLik = logLik + logLikIncrement;

        % Update + renormalise weights
        logW = logW + logW_inc - logLikIncrement;

        % Filtered-state mean (linear-space expectation)
        if opts.returnPath
            hFiltered(t) = sum(exp(logW) .* h);
        end

        % Resample if ESS too low (skip the final step to avoid noise
        % inflation when no future propagation will use the indices).
        if t < T
            logESS = -utils.logsumexp(2 * logW);
            if logESS < log(opts.essThreshold * M)
                idx  = resampler(logW);
                h    = h(idx);
                aux  = reindexAux(aux, idx);
                logW = -log(M) * ones(M, 1);
            end
        end
    end
end


function aux = reindexAux(aux, idx)
% reindexAux  Apply resampling indices to every per-particle field of aux.
%   Scalar (non-particle) fields are passed through. Vector fields whose
%   length matches the number of indices are reordered. Anything else is
%   left as-is, on the assumption that the model authored aux knows what
%   it is doing.
    fn = fieldnames(aux);
    M  = numel(idx);
    for k = 1:numel(fn)
        v = aux.(fn{k});
        if isnumeric(v) && isvector(v) && numel(v) == M
            aux.(fn{k}) = v(idx);
        end
    end
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
