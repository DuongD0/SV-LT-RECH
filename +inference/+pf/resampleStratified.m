function idx = resampleStratified(logW)
% resampleStratified  Stratified resampling from normalised log-weights.
%
%   idx = resampleStratified(logW) -> N-by-1 ancestor index vector.
%
%   N independent uniform draws on (k/N, (k+1)/N], k = 0..N-1. Variance
%   sits between multinomial (worst) and systematic (best). We keep it
%   available because some convergence diagnostics depend on independent
%   resampling noise across particles.

    N = numel(logW);
    logW = logW - utils.logsumexp(logW(:));
    w = exp(logW(:));

    u = (rand(N, 1) + (0:N-1)') / N;

    cumW = cumsum(w);
    cumW(end) = 1;
    idx = zeros(N, 1);
    j = 1;
    for i = 1:N
        while u(i) > cumW(j)
            j = j + 1;
        end
        idx(i) = j;
    end
end
