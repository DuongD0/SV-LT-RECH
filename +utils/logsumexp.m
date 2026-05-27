function y = logsumexp(x, dim)
% logsumexp  Numerically stable log(sum(exp(x))).
%
%   y = logsumexp(x)         operates on the first non-singleton dim.
%   y = logsumexp(x, dim)    operates along dimension dim.
%
%   Subtracts the maximum before exponentiating so overflow / underflow
%   never reach the exp() call. Returns -Inf for an all-(-Inf) slice.

    if nargin < 2
        dim = find(size(x) > 1, 1);
        if isempty(dim); dim = 1; end
    end

    m = max(x, [], dim);
    finiteMask = isfinite(m);

    shifted = x - m;
    s = sum(exp(shifted), dim);
    y = m + log(s);

    %% Carry -Inf through cleanly when an entire slice is -Inf
    y(~finiteMask) = m(~finiteMask);
end
