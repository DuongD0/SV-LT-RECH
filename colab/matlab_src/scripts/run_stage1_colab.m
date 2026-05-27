%% run_stage1_colab  Entry script for the Colab Stage 1 study.
%
%   Run from MATLAB on Colab as:
%       matlab -batch "addpath('matlab_src'); run('matlab_src/scripts/run_stage1_colab.m')"
%
%   Or from a Colab Python cell as:
%       !cd /content/finance_eng_colab && matlab -batch \
%           "addpath('matlab_src'); run('matlab_src/scripts/run_stage1_colab.m')"
%
%   The runner is resumable; if the Colab VM expires mid-run, just
%   re-launch and it picks up from the last completed replicate.
%
%   Two configs are bundled (JSON; YAML twins kept for human reading):
%     stage1_colab.json       full publication-quality study (multi-session)
%     stage1_colab_quick.json single-session sanity (~30 min)
%
%   Defaults to the QUICK variant. To run the full study, set the
%   environment variable STAGE1_CONFIG=stage1_colab.json before
%   launching MATLAB, or edit the configName line below.

%% Make `addpath('matlab_src')` work regardless of the caller's cwd.
%% This script lives at matlab_src/scripts/, so matlab_src/ is one
%% level up and the bundle root is two levels up. Anchor everything
%% (paths AND default outDir) to the bundle root so outputs land next
%% to the notebook regardless of MATLAB's cwd at launch.
scriptDir   = fileparts(mfilename('fullpath'));
matlabSrc   = fileparts(scriptDir);                  % .../matlab_src
bundleRoot  = fileparts(matlabSrc);                  % .../  (colab bundle)
addpath(genpath(matlabSrc));

configName = getenv('STAGE1_CONFIG');
if isempty(configName)
    configName = 'stage1_colab_quick.json';
end
if endsWith(configName, '.yaml')
    configName = [configName(1:end-5), '.json'];
end

configPath = fullfile(matlabSrc, 'config', 'experiments', configName);
fprintf('=== run_stage1_colab: %s ===\n', configPath);

outDir = getenv('STAGE1_OUTDIR');
if isempty(outDir)
    outDir = fullfile(bundleRoot, 'results', erase(configName, '.json'));
end

opts.outDir       = outDir;
opts.writeOutputs = true;
opts.resume       = true;

out = experiments.runStage1Colab(string(configPath), opts);

fprintf('\n=== Stage 1 study done ===\n');
fprintf('coverage95 per param: %s\n', mat2str(out.summary.coverage95, 3));
fprintf('max R-hat:            %.3f\n', out.summary.maxRhat);
fprintf('MCSE log Z (max):     %.3f\n', out.summary.mcseLogZ_max);
fprintf('max sign-flip frac:   %.3f\n', out.summary.maxFracFlipped);
fprintf('Output bundle:        %s\n', outDir);
fprintf('Gate verdict file:    %s/gate_summary.json\n', outDir);
