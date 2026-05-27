function lp = logPriorSVLTGRURECH(theta, nCovariates)
% logPriorSVLTGRURECH  Log prior for SV-LT-GRU-RECH (PROPOSED_METHODOLOGY.md §8.5).
%
%   lp = logPriorSVLTGRURECH(theta, nCovariates)
%
%   theta order (length 13 + 3*nInputs, nInputs = 3 + nCovariates):
%     [mu, phi, sigma_eta, rho, nu, beta_0, beta_1,
%      <gate z: W_z(1..nInputs), u_z, b_z>,
%      <gate r: W_r(1..nInputs), u_r, b_r>,
%      <gate h: W_h(1..nInputs), u_h, b_h>]
%
%   Priors (PROPOSED_METHODOLOGY.md §8.5 + §9.1)
%   -------------------------------------------
%     mu        ~ N(0, 10^2)
%     (1+phi)/2 ~ Beta(20, 1.5)
%     sigma_eta ~ Half-Cauchy(0, 1)
%     rho       ~ Uniform(-1, 1)
%     nu        ~ Gamma(1, 0.1) truncated to nu > 2
%     beta_0    ~ Uniform(0, 0.5)
%     beta_1    ~ Uniform(0, 0.5)
%     all GRU gate weights ~ N(0, 0.1^2)
%
%   Returns -Inf if any parameter is out of support.

    arguments
        theta       (1,:) double
        nCovariates (1,1) double {mustBeNonnegative}
    end

    K        = nCovariates;
    nInputs  = 3 + K;
    nGateW   = 3 * (nInputs + 2);          % 3 gates x (nInputs W + u + b)
    expected = 7 + nGateW;
    assert(numel(theta) == expected, ...
        'logPriorSVLTGRURECH:badLength', ...
        'theta length %d does not match expected %d for nCovariates=%d', ...
        numel(theta), expected, K);

    phi       = theta(2);
    sigma_eta = theta(3);
    rho       = theta(4);
    nu        = theta(5);
    beta_0    = theta(6);
    beta_1    = theta(7);
    gateW     = theta(8:end);

    if phi <= -1 || phi >= 1 || sigma_eta <= 0 || nu <= 2 ...
            || rho <= -1 || rho >= 1 ...
            || beta_0 < 0 || beta_0 > 0.5 ...
            || beta_1 < 0 || beta_1 > 0.5
        lp = -Inf;
        return
    end

    lp_core = priors.logPriorSVLT(theta(1:5));

    % Uniform(0, 0.5) on beta_0, beta_1.
    lp_beta = -2 * log(0.5);

    % N(0, 0.1^2) on every GRU gate weight.
    sd     = 0.1;
    nW     = numel(gateW);
    lp_rnn = -0.5 * sum(gateW.^2) / sd^2 ...
             - nW * (log(sd) + 0.5 * log(2 * pi));

    lp = lp_core + lp_beta + lp_rnn;
end
