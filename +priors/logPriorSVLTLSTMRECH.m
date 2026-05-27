function lp = logPriorSVLTLSTMRECH(theta, nCovariates)
% logPriorSVLTLSTMRECH  Log prior for SV-LT-LSTM-RECH (PROPOSED_METHODOLOGY.md §8.4).
%
%   lp = logPriorSVLTLSTMRECH(theta, nCovariates)
%
%   theta order (length 15 + 4*nInputs, nInputs = 3 + nCovariates):
%     [mu, phi, sigma_eta, rho, nu, beta_0, beta_1,
%      <gate f: W_f(1..nInputs), u_f, b_f>,
%      <gate i: W_i(1..nInputs), u_i, b_i>,
%      <gate o: W_o(1..nInputs), u_o, b_o>,
%      <gate c: W_c(1..nInputs), u_c, b_c>]
%
%   Priors (PROPOSED_METHODOLOGY.md §8.4 + §9.1)
%   -------------------------------------------
%     mu        ~ N(0, 10^2)
%     (1+phi)/2 ~ Beta(20, 1.5)
%     sigma_eta ~ Half-Cauchy(0, 1)
%     rho       ~ Uniform(-1, 1)
%     nu        ~ Gamma(1, 0.1) truncated to nu > 2
%     beta_0    ~ Uniform(0, 0.5)
%     beta_1    ~ Uniform(0, 0.5)
%     all LSTM gate weights ~ N(0, 0.1^2)        (§8.4: "all weights N(0, 0.1)")
%
%   Returns -Inf if any parameter is out of support.

    arguments
        theta       (1,:) double
        nCovariates (1,1) double {mustBeNonnegative}
    end

    K        = nCovariates;
    nInputs  = 3 + K;
    nGateW   = 4 * (nInputs + 2);          % 4 gates x (nInputs W + u + b)
    expected = 7 + nGateW;
    assert(numel(theta) == expected, ...
        'logPriorSVLTLSTMRECH:badLength', ...
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

    % N(0, 0.1^2) on every LSTM gate weight.
    sd     = 0.1;
    nW     = numel(gateW);
    lp_rnn = -0.5 * sum(gateW.^2) / sd^2 ...
             - nW * (log(sd) + 0.5 * log(2 * pi));

    lp = lp_core + lp_beta + lp_rnn;
end
