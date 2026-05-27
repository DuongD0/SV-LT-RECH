%% run_stage1_simulation  Stage 1 simulation study driver (§11.1).
%
%   Manual entry point for the FULL replicate study. Delegates to
%   `experiments.runStage1` so the test harness can call the same body
%   on the smoke config.
%
%   Default config:  config/experiments/stage1.yaml
%
%   Outputs (under results/stage1/):
%     recovery.csv      cols: replicate, seed, param, true, postMean,
%                              postStd, withinCI95, rHat
%     mcse_logml.csv    cols: logZ_mean, logZ_mcse, replicate
%     sign_switch.csv   cols: replicate, seed, fracFlipped
%     figures/F11_replicate1_posterior.png
%
%   Acceptance gates are in phase log §4.5; this script writes the raw
%   materials. Gate evaluation happens in PHASE_3_COMPLETE.md.

addpath(pwd);
configPath = fullfile('config', 'experiments', 'stage1.yaml');
out = experiments.runStage1(string(configPath));

fprintf('[stage1] coverage95: %s\n', mat2str(out.summary.coverage95, 3));
fprintf('[stage1] max R-hat:  %.3f\n', out.summary.maxRhat);
fprintf('[stage1] MCSE logZ:  %.3f\n', out.summary.mcseLogZ_max);
fprintf('[stage1] max sign-flip frac: %.3f\n', out.summary.maxFracFlipped);
