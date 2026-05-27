function idx = resampleSystematic(logW)
% resampleSystematic  Systematic resampling from normalised log-weights.
%
%   idx = resampleSystematic(logW) -> N-by-1 ancestor index vector in [1, N].
%
%   logW need not be normalised; we normalise internally with logsumexp.
%
%   Reference: Doucet, de Freitas & Gordon (2001), p. 13. Systematic
%   resampling has lower variance than multinomial and stratified
%   resampling for the same N.

    N = numel(logW);
    logW = logW - utils.logsumexp(logW(:));
    w = exp(logW(:));

    %% Single uniform draw; deterministic offsets thereafter
    u = (rand() + (0:N-1)') / N;

    cumW = cumsum(w);
    cumW(end) = 1;                              %#ok<NASGU>  guard FP drift
    idx = zeros(N, 1);
    j = 1;
    for i = 1:N
        while u(i) > cumW(j)
            j = j + 1;
        end
        idx(i) = j;
    end
end
