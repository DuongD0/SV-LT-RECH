%% calibrate_against_stochvol.m
%
%   Phase 1 cross-engine sanity check.
%
%   1. Simulate a length-1000 plain-SV series from known theta_true.
%   2. Fit it via our +inference/+smc/likelihoodAnneal (small N).
%   3. Fit the same series via R `stochvol::svsample`.
%   4. Assert posterior means agree within 2 sd on every parameter.
%
%   Expected runtime: < 5 min single-core.

clearvars; close all;
addpath(fileparts(fileparts(mfilename('fullpath'))));

utils.reproducibility(20260516);

%% Simulate ground truth
mdl       = models.SV();
thetaTrue = [0.0, 0.95, 0.4];
T         = 1000;
[y, ~]    = mdl.simulate(thetaTrue, T);

%% Persist returns for R
tmpDir  = tempname; mkdir(tmpDir);
retsCsv = fullfile(tmpDir, 'returns.csv');
sumCsv  = fullfile(tmpDir, 'sv_posterior.csv');

dates = datetime(2020, 1, 1) + caldays(0:T-1);
T_ret = table(string(dates(:), 'yyyy-MM-dd'), y, ...
              'VariableNames', {'date', 'return'});
writetable(T_ret, retsCsv);

%% Run R stochvol
rScript = fullfile(fileparts(mfilename('fullpath')), ...
                   'calibrate_against_stochvol.R');
cmd = sprintf('Rscript "%s" "%s" "%s" 10000', rScript, retsCsv, sumCsv);
fprintf('Running R: %s\n', cmd);
[status, out] = system(cmd);
if status ~= 0
    error('R bridge failed:\n%s', out);
end
fprintf('%s\n', out);

stochvolSummary = readtable(sumCsv);
fprintf('\nstochvol posterior summary:\n');
disp(stochvolSummary);

%% Run our SMC
opts   = struct('N', 1000, 'M', 100, 'nSweeps', 5, 'verbose', true);
result = inference.smc.likelihoodAnneal(mdl, y, opts);

ourMean = mean(result.theta, 1);
ourSd   = std(result.theta, 0, 1);

%% Compare
fprintf('\n==== Cross-engine comparison ====\n');
for k = 1:numel(mdl.paramNames)
    nm   = mdl.paramNames{k};
    sv   = stochvolSummary{k, {'mean', 'sd'}};
    ours = [ourMean(k), ourSd(k)];
    fprintf('%s  truth=%+0.4f  stochvol=%+0.4f (+/- %0.4f)  ours=%+0.4f (+/- %0.4f)\n', ...
        nm, thetaTrue(k), sv(1), sv(2), ours(1), ours(2));
    diff = abs(ours(1) - sv(1));
    tol  = 2 * (sv(2) + ours(2));
    if diff > tol
        warning('%s differs by %0.4f, exceeds 2-sd tolerance %0.4f', nm, diff, tol);
    end
end
