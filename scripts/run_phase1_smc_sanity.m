%% run_phase1_smc_sanity.m
%
%   Phase 1 smoke test:
%     1. Run every unit test under tests/.
%     2. Run the cross-engine plain-SV calibration vs R stochvol.
%
%   Errors out if any test fails.

clearvars; close all;
addpath(fileparts(fileparts(mfilename('fullpath'))));

utils.reproducibility(20260516);

%% Step 1: unit tests
fprintf('\n==== Phase 1 unit tests ====\n');
results = runtests('tests', 'IncludeSubfolders', true);
disp(results);
if any([results.Failed])
    error('Unit tests failed; fix before proceeding.');
end

%% Step 2: cross-engine calibration (skips if R missing)
fprintf('\n==== Cross-engine SV calibration ====\n');
[hasR, ~] = system('which Rscript');
if hasR ~= 0
    warning('Rscript not available; skipping cross-engine calibration.');
else
    run(fullfile(fileparts(mfilename('fullpath')), ...
                 'calibrate_against_stochvol.m'));
end

fprintf('\nPhase 1 sanity complete.\n');
