function reg = stage4Models()
% stage4Models  Registry of the Stage-4 comparison models.
%
%   reg = experiments.stage4Models()
%
%   Single source of truth for which models the Stage-4 pipeline fits and
%   reports. The set spans a symmetric {backbone} x {deep-learning} grid so
%   the effectiveness of the RNN augmentation can be read off directly:
%
%                        no deep learning        + deep learning (SRN/LSTM/GRU)
%     GARCH backbone     GARCH-t, GJR-t          GARCH-RECH-{SRN,LSTM,GRU}
%     SV    backbone     SV, SVM, SVLT           SVLTRECH-{SRN,LSTM,GRU}
%
%   plus SVM (stochastic volatility IN MEAN) for the risk-return trade-off.
%
%   Each entry:
%     .name       display label
%     .kind       'garch' (Econometrics MLE) | 'garchrech' (RECH MLE) | 'sv' (SMC)
%     .garchType  'garch' | 'gjr'   (kind=='garch' only; '' otherwise)
%     .cell       'srn' | 'lstm' | 'gru'  (kind=='garchrech' only; '' otherwise)
%     .ctor       @(K, leverage) -> models.Model  (kind=='sv' only; [] else)
%     .isProposed true for the flagship vs which baselines are DM-compared
%
%   See PROPOSED_METHODOLOGY.md §7-8 and the Phase-6 design spec §4.2.

    %% --- GARCH backbone: classic (no DL) ---
    reg        = entry('GARCH-t',         'garch',     'garch', '',     [], false);
    reg(end+1) = entry('GJR-t',           'garch',     'gjr',   '',     [], false);

    %% --- GARCH backbone: deep learning (original RECH, Nguyen-Tran-Kohn 2022) ---
    reg(end+1) = entry('GARCH-RECH-SRN',  'garchrech', '',      'srn',  [], false);
    reg(end+1) = entry('GARCH-RECH-LSTM', 'garchrech', '',      'lstm', [], false);
    reg(end+1) = entry('GARCH-RECH-GRU',  'garchrech', '',      'gru',  [], false);

    %% --- SV backbone: no deep learning ---
    reg(end+1) = entry('SV',              'sv', '', '', @(~, ~)   models.SV(),                                       false);
    reg(end+1) = entry('SVM',             'sv', '', '', @(~, ~)   models.SVM(),                                      false);
    reg(end+1) = entry('SVLT',            'sv', '', '', @(~, ~)   models.SVLT(),                                     false);

    %% --- SV backbone: deep learning (flagship = SVLTRECH-SRN) ---
    reg(end+1) = entry('SVLTRECH-SRN',    'sv', '', '', @(K, lev) models.SVLTRECH('nCovariates', K, 'leverage', lev),     true);
    reg(end+1) = entry('SVLTRECH-LSTM',   'sv', '', '', @(K, lev) models.SVLTLSTMRECH('nCovariates', K, 'leverage', lev), false);
    reg(end+1) = entry('SVLTRECH-GRU',    'sv', '', '', @(K, lev) models.SVLTGRURECH('nCovariates', K, 'leverage', lev),  false);
end


function e = entry(name, kind, garchType, cell, ctor, isProposed)
    e = struct('name', name, 'kind', kind, 'garchType', garchType, ...
               'cell', cell, 'ctor', ctor, 'isProposed', isProposed);
end
