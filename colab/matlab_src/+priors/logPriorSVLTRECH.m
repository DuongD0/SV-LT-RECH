function lp = logPriorSVLTRECH(theta, nCovariates)
% logPriorSVLTRECH  Log prior for SV-LT-RECH (PROPOSED_METHODOLOGY.md §8.2).
%
%   lp = logPriorSVLTRECH(theta, nCovariates)
%
%   theta order (length 12 + nCovariates):
%     [mu, phi, sigma_eta, rho, nu, beta_0, beta_1,
%      v_h, v_r, v_omega, v_z(1)..v_z(K), w_h, b]
%
%   Priors (PROPOSED_METHODOLOGY.md §9.1 + phase log §4.2.3)
%   --------------------------------------------------------
%     mu        ~ N(0, 10^2)
%     (1+phi)/2 ~ Beta(20, 1.5)
%     sigma_eta ~ Half-Cauchy(0, 1)
%     rho       ~ Uniform(-1, 1)
%     nu        ~ Gamma(1, 0.1) truncated to nu > 2
%     beta_0    ~ Uniform(0, 0.5)
%     beta_1    ~ Uniform(0, 0.5)
%     v_h, v_r, v_omega ~ N(0, 0.1^2)
%     v_z       ~ N(0, 0.5^2)        (one per covariate)
%     w_h, b    ~ N(0, 0.1^2)
%
%   Returns -Inf if any parameter is out of support.

    arguments
        theta       (1,:) double
        nCovariates (1,1) double {mustBeNonnegative}
    end

    K        = nCovariates;
    expected = 12 + K;
    assert(numel(theta) == expected, ...
        'logPriorSVLTRECH:badLength', ...
        'theta length %d does not match expected %d for nCovariates=%d', ...
        numel(theta), expected, K);

    mu        = theta(1);
    phi       = theta(2);
    sigma_eta = theta(3);
    rho       = theta(4);
    nu        = theta(5);
    beta_0    = theta(6);
    beta_1    = theta(7);
    v_h       = theta(8);
    v_r       = theta(9);
    v_omega   = theta(10);
    if K > 0
        v_z = theta(11 : 10 + K);
    else
        v_z = [];
    end
    w_h = theta(11 + K);
    b   = theta(12 + K);

    if phi <= -1 || phi >= 1 || sigma_eta <= 0 || nu <= 2 ...
            || rho <= -1 || rho >= 1 ...
            || beta_0 < 0 || beta_0 > 0.5 ...
            || beta_1 < 0 || beta_1 > 0.5
        lp = -Inf;
        return
    end

    lp_core = priors.logPriorSVLT([mu, phi, sigma_eta, rho, nu]);

    % Uniform(0, 0.5) on beta_0, beta_1: log density = -log(0.5) each.
    lp_beta = -2 * log(0.5);

    % N(0, 0.1^2) on v_h, v_r, v_omega, w_h, b   (5 scalar weights).
    sdSmall = 0.1;
    lp_rnn  = -0.5 * (v_h^2 + v_r^2 + v_omega^2 + w_h^2 + b^2) / sdSmall^2 ...
              - 5 * (log(sdSmall) + 0.5 * log(2 * pi));

    % N(0, 0.5^2) on v_z (one per covariate).
    sdCov = 0.5;
    lp_z  = -0.5 * sum(v_z.^2) / sdCov^2 ...
            - K * (log(sdCov) + 0.5 * log(2 * pi));

    lp = lp_core + lp_beta + lp_rnn + lp_z;
end
