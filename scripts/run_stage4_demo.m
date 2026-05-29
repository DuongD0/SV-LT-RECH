%% run_stage4_demo  Offline end-to-end demo of the Phase-6 reporting chain.
%  Synthetic VN-like series -> 6-model Stage-4 fit -> bundle -> tables + figures.
%  No network, small N/M/J. Run:
%    matlab -batch "addpath(pwd); addpath('scripts'); run('scripts/run_stage4_demo.m')"

utils.reproducibility(20260528);

K        = 2;
T        = 320;
splitIdx = 240;
thetaTrue = [-0.5, 0.95, 0.25, -0.4, 8, 0.05, 0.20, ...
             0.05, -0.05, 0.05, 0.10, -0.10, 0.10, 0.0];   % 12 + 2
Z   = randn(T, K);
gen = models.SVLTRECH('nCovariates', K, 'leverage', 'cholesky');
gen.setCovariates(Z);
[y, ~] = gen.simulate(thetaTrue, T);

cfg = struct();
cfg.model    = struct('nCovariates', K, 'leverage', 'cholesky');
cfg.smc      = struct('N', 300, 'M', 40, 'nSweeps', 3, 'proposalScale', 0.5, ...
                      'essThreshold', 0.5, 'targetEss', 0.5, 'verbose', false);
cfg.forecast = struct('J', 30);
cfg.eval     = struct('alphaQS', [0.01, 0.05], 'mcsB', 500);
cfg.particleFilter = struct('clipLogWeight', -50);
cfg.baseSeed = 20260528;

% Anchor outputs to the project root: MATLAB's run() cd's into scripts/, so a
% bare relative 'results/...' would land under scripts/. Derive the root from
% this file's location instead.
projRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(projRoot, 'results', 'stage4_demo');
out = experiments.runStage4(y, Z, splitIdx, cfg, ...
    struct('writeOutputs', true, 'outDir', outDir, 'expName', 'stage4_demo'));

make_paper_tables(outDir);

fprintf('\n[demo] models: %s\n', strjoin(out.modelNames, ', '));
for m = 1:numel(out.modelNames)
    s = out.scores(m);
    fprintf('  %-14s QLIKE=%.3f PPS=%.3f inMCS=%d\n', ...
        out.modelNames{m}, s.qlike, s.pps, out.mcs.inSet(m));
end
fprintf('[demo] tables + figures in %s\n', outDir);
