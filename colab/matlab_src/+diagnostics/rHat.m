function rhat = rHat(thetaChains)
% rHat  Gelman-Rubin potential scale reduction factor (split R-hat).
%
%   rhat = rHat(thetaChains)
%
%   Input
%   -----
%     thetaChains : N-by-K-by-C array of posterior draws
%                   (N particles per chain, K parameters, C chains).
%
%   Output
%   ------
%     rhat : 1-by-K vector of split-R-hat values. Values near 1.0
%            indicate between-chain agreement; > 1.1 flags poor mixing.
%
%   Uses split R-hat (Gelman et al. 2014 BDA ch.11): each of the C
%   chains is split in half along the particle dimension, giving 2C
%   effective chains. Returns NaN when N < 4 or C < 2.

    arguments
        thetaChains (:,:,:) double
    end

    [N, K, C] = size(thetaChains);
    if N < 4 || C < 2
        rhat = nan(1, K);
        return
    end

    half = floor(N / 2);
    split = zeros(half, K, 2 * C);
    for c = 1:C
        split(:, :, 2*c - 1) = thetaChains(1:half,          :, c);
        split(:, :, 2*c    ) = thetaChains(half+1 : 2*half, :, c);
    end

    M = 2 * C;
    n = half;

    chainMeans = reshape(mean(split, 1), [K, M])';   % M-by-K
    chainVars  = reshape(var(split, 0, 1), [K, M])';

    overallMean = mean(chainMeans, 1);
    B = n * sum((chainMeans - overallMean).^2, 1) / (M - 1);
    W = mean(chainVars, 1);

    varHat = (n - 1) / n * W + B / n;
    rhat   = sqrt(varHat ./ max(W, eps));
end
