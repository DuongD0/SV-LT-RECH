function reg = stage4Models()
% stage4Models  Registry of the six Stage-4 comparison models.
%
%   reg = experiments.stage4Models()
%
%   Single source of truth for which models the Stage-4 pipeline fits and
%   reports. Each entry:
%     .name       display label
%     .kind       'garch' (Econometrics-Toolbox MLE) | 'sv' (SMC)
%     .garchType  'garch' | 'gjr'   (kind=='garch' only; '' otherwise)
%     .ctor       @(K, leverage) -> models.Model  (kind=='sv' only; [] else)
%     .isProposed true for the flagship vs which baselines are DM-compared
%
%   See PROPOSED_METHODOLOGY.md §7-8 and the Phase-6 design spec §4.2.

    reg        = entry('GARCH-t',       'garch', 'garch', [], false);
    reg(end+1) = entry('GJR-t',         'garch', 'gjr',   [], false);
    reg(end+1) = entry('SVLT',          'sv', '', @(~, ~)    models.SVLT(),                                      false);
    reg(end+1) = entry('SVLTRECH-SRN',  'sv', '', @(K, lev)  models.SVLTRECH('nCovariates', K, 'leverage', lev),     true);
    reg(end+1) = entry('SVLTRECH-LSTM', 'sv', '', @(K, lev)  models.SVLTLSTMRECH('nCovariates', K, 'leverage', lev), false);
    reg(end+1) = entry('SVLTRECH-GRU',  'sv', '', @(K, lev)  models.SVLTGRURECH('nCovariates', K, 'leverage', lev),  false);
end


function e = entry(name, kind, garchType, ctor, isProposed)
    e = struct('name', name, 'kind', kind, 'garchType', garchType, ...
               'ctor', ctor, 'isProposed', isProposed);
end
