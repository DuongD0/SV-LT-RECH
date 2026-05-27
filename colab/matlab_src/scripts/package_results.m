%% package_results  Bundle Stage 1 Colab outputs for download.
%
%   Run AFTER the Stage 1 study completes:
%       matlab -batch "run('matlab_src/scripts/package_results.m')"
%
%   Reads:
%     results/<runId>/recovery.csv
%     results/<runId>/mcse_logml.csv
%     results/<runId>/sign_switch.csv
%     results/<runId>/gate_summary.json
%     results/<runId>/posteriors/*.mat
%     results/<runId>/simulations/*.mat
%     results/<runId>/figures/*.png
%
%   Writes:
%     phase4_handoff_<runId>_<timestamp>.tar.gz
%
%   This single tarball is what you ship back to the assistant for
%   Phase 4 ingestion.

configName = getenv('STAGE1_CONFIG');
if isempty(configName)
    configName = 'stage1_colab_quick.json';
end
if endsWith(configName, '.yaml')
    configName = [configName(1:end-5), '.json'];
end
runId = erase(configName, '.json');

%% Anchor to the bundle root, same convention as run_stage1_colab.m.
scriptDir  = fileparts(mfilename('fullpath'));
matlabSrc  = fileparts(scriptDir);
bundleRoot = fileparts(matlabSrc);
resultsDir = fullfile(bundleRoot, 'results', runId);
if ~isfolder(resultsDir)
    error('package_results:noResults', ...
          'Results directory not found: %s', resultsDir);
end

timestamp   = datestr(now, 'yyyymmdd_HHMMSS');
archiveName = sprintf('phase4_handoff_%s_%s.tar.gz', runId, timestamp);
archivePath = fullfile(pwd, archiveName);

manifest = struct();
manifest.config_used  = configName;
manifest.results_dir  = resultsDir;
manifest.archive_path = archivePath;
manifest.packaged_at  = timestamp;

if isfile(fullfile(resultsDir, 'gate_summary.json'))
    fid  = fopen(fullfile(resultsDir, 'gate_summary.json'), 'r');
    gate = char(fread(fid)');
    fclose(fid);
    manifest.gate_summary_inline = jsondecode(gate);
end

fid = fopen(fullfile(resultsDir, 'MANIFEST.json'), 'w');
fwrite(fid, jsonencode(manifest, 'PrettyPrint', true));
fclose(fid);

cmd = sprintf('tar -czvf "%s" -C "%s" .', archivePath, resultsDir);
fprintf('[package] %s\n', cmd);
system(cmd);

info = dir(archivePath);
fprintf('\n=== Bundle ready ===\n');
fprintf('Path:  %s\n', archivePath);
fprintf('Size:  %.2f MB\n', info.bytes / 1024 / 1024);
fprintf('\nDownload from Colab:\n');
fprintf('  from google.colab import files\n');
fprintf('  files.download(''%s'')\n', archivePath);
