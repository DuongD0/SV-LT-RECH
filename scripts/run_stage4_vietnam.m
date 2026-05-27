%% run_stage4_vietnam  Stage 4 entry point (PROPOSED_METHODOLOGY §11.4).
%  Run after `source setup.sh`:
%    matlab -batch "addpath(pwd); pyenv('Version','.venv/bin/python'); run('scripts/run_stage4_vietnam.m')"

cfg = utils.loadConfig('config/smc_defaults.yaml', 'config/experiments/stage4.yaml');

%% Fetch + align
ds = data.loadStage4(cfg);
fprintf('[stage4] loaded %d obs, %d covariates (%s)\n', ...
        numel(ds.y), numel(ds.covNames), strjoin(ds.covNames, ', '));

%% Fetch-sanity anchor (NEU thesis Table 4.14: kurtosis ~7.73, std ~0.498)
fprintf('[stage4] VN-Index returns: mean=%.4f std=%.4f skew=%.4f kurt=%.4f n=%d\n', ...
        mean(ds.y), std(ds.y), skewness(ds.y), kurtosis(ds.y), numel(ds.y));

%% Leakage-safe covariate standardisation using TRAIN stats only
ZTrain = ds.Z(1:ds.splitIdx, :);
ZTest  = ds.Z(ds.splitIdx+1:end, :);
[ZTrainStd, ZTestStd] = data.preprocess.standardize(ZTrain, ZTest);
Zstd = [ZTrainStd; ZTestStd];

%% Run the pipeline
out = experiments.runStage4(ds.y, Zstd, ds.splitIdx, cfg, ...
        struct('writeOutputs', true, 'outDir', fullfile('results','stage4')));

%% Report
fprintf('\n[stage4] scores written to results/stage4/scores.csv\n');
for m = 1:numel(out.modelNames)
    s = out.scores(m);
    fprintf('  %-9s  PPS=%.3f QLIKE=%.3f MSE=%.3f MAE=%.3f R2LOG=%.3f  inMCS=%d\n', ...
        out.modelNames{m}, s.pps, s.qlike, s.mse, s.mae, s.r2log, out.mcs.inSet(m));
end
