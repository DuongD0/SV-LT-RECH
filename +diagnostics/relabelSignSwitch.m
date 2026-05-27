function [thetaOut, report] = relabelSignSwitch(theta, beta1Idx)
% relabelSignSwitch  Detect and (defensively) clamp beta_1 sign symmetry.
%
%   [thetaOut, report] = relabelSignSwitch(theta, beta1Idx)
%
%   The RECH cell s_t = ReLU(...) has a sign symmetry only when
%   beta_1 = 0, which the prior already excludes via Uniform(0, 0.5).
%   This diagnostic verifies that the prior held during SMC: any
%   posterior particle with beta_1 < 0 signals a prior-violation bug
%   in the RWM kernel or a numerical leak through the support check.
%
%   Inputs
%   ------
%     theta    : N-by-K matrix of posterior particles.
%     beta1Idx : scalar column index of beta_1 in theta.
%
%   Outputs
%   -------
%     thetaOut : N-by-K, with beta_1 column clamped to >= 0 (defensive).
%     report   : struct
%                  .nFlipped       number of particles with beta_1 < 0
%                  .fracFlipped    fraction in [0, 1]
%                  .beta1Quantiles 5/50/95 quantiles of beta_1 (pre-clamp)

    arguments
        theta    (:,:) double
        beta1Idx (1,1) double {mustBePositive, mustBeInteger}
    end

    beta1 = theta(:, beta1Idx);
    mask  = beta1 < 0;

    thetaOut = theta;
    thetaOut(mask, beta1Idx) = 0;

    report = struct( ...
        'nFlipped',       sum(mask), ...
        'fracFlipped',    mean(mask), ...
        'beta1Quantiles', quantile(beta1, [0.05, 0.50, 0.95]));
end
