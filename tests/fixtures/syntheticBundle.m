function bundle = syntheticBundle()
% syntheticBundle  Small in-memory results bundle matching runStage4's schema.
%   Two SV models + two GARCH models, K=2 covariates, T=60, split=40.
%   Deterministic (fixed rng) so report tests are reproducible.

    rng(2026, 'threefry');
    T = 60; splitIdx = 40; K = 2;
    y = 0.5 * randn(T, 1);
    Z = randn(T, K);
    testIdx = (splitIdx + 1 : T)';
    nTest   = numel(testIdx);

    names = {'GARCH-t','GJR-t','SVLTRECH-SRN','SVLTRECH-LSTM'};
    nM    = numel(names);

    models_ = repmat(modelRec(), 1, nM);
    for m = 1:nM
        r = modelRec();
        r.name = names{m};
        r.nu   = 6 + m;
        r.varForecast    = 0.2 + 0.05 * rand(nTest, 1);
        r.logPredDensity = -1 - 0.1 * rand(nTest, 1);
        r.sigma2InSample = 0.2 + 0.05 * rand(splitIdx, 1);
        r.stdResid       = randn(splitIdx, 1);
        if startsWith(names{m}, 'SVLTRECH')
            r.type           = 'sv';
            r.logMarginalLik = -100 - m;
            r.theta          = [randn(80, 5), 6 + randn(80, 1), ...
                                0.1 * randn(80, 6 + K)];     % 12 + K cols
            r.paramNames     = svParamNames(K);
            r.posteriorMean  = mean(r.theta, 1);
            r.posteriorStd   = std(r.theta, 0, 1);
            r.omegaPath      = 0.05 + 0.01 * randn(splitIdx, 1);
        else
            r.type           = 'garch';
            r.logMarginalLik = NaN;
            r.paramNames     = {'omega','alpha','beta','nu'};
            r.posteriorMean  = [0.02, 0.08, 0.88, r.nu];
            r.posteriorStd   = nan(1, 4);
            r.theta          = [];
            r.omegaPath      = [];
        end
        models_(m) = r;
    end

    qlAll = 0.5 + 0.1 * rand(nTest, nM);
    scores = repmat(scoreRec(), 1, nM);
    for m = 1:nM
        scores(m) = struct('pps', 1 + 0.1*m, 'qs1', 0.03 + 0.001*m, ...
            'qs5', 0.05 + 0.001*m, 'mse', 0.1 + 0.01*m, 'mae', 0.2 + 0.01*m, ...
            'r2log', 0.7 + 0.05*m, 'qlike', -1 + 0.05*m);
    end
    dm = repmat(struct('statistic', 1.2, 'pValue', 0.2, 'meanDiff', 0.01, ...
                       'bandwidth', 3), 1, nM - 1);
    mcs = struct('inSet', [false, false, true, true], ...
                 'pValues', [0.02, 0.04, 0.9, 0.6], ...
                 'eliminationOrder', [1 2 4 3], 'statistic', 'Tmax');

    bundle.meta = struct('expName', 'synthetic', 'date', '2026-05-28', ...
                         'baseSeed', 2026, 'modelNames', {names});
    bundle.data = struct('y', y, 'Z', Z, 'splitIdx', splitIdx, ...
                         'testIdx', testIdx, 'nCovariates', K);
    bundle.descriptives = struct('n', splitIdx, 'mean', mean(y(1:splitIdx)), ...
        'std', std(y(1:splitIdx)), 'skewness', skewness(y(1:splitIdx)), ...
        'kurtosis', kurtosis(y(1:splitIdx)), 'min', min(y), 'max', max(y), ...
        'adf', pv(0.01), 'pp', pv(0.01), 'kpss', pv(0.1), ...
        'jb', pv(0.001), 'archlm', pv(0.001), ...
        'bds', struct('stat', 6.2, 'pValue', 1e-9, 'm', 2, 'eps', 0.25));
    bundle.models     = models_;
    bundle.comparison = struct('modelNames', {names}, 'scores', scores, ...
        'qlAll', qlAll, 'dmVsProposed', dm, 'dmBaselines', {names(1:end-1)}, ...
        'mcs', mcs, 'proposedName', 'SVLTRECH-SRN');
end

function r = modelRec()
    r = struct('name','','type','','nu',NaN,'varForecast',[],'logPredDensity',[], ...
        'logMarginalLik',NaN,'sigma2InSample',[],'stdResid',[],'omegaPath',[], ...
        'theta',[],'paramNames',{{}},'posteriorMean',[],'posteriorStd',[]);
end
function s = scoreRec()
    s = struct('pps',NaN,'qs1',NaN,'qs5',NaN,'mse',NaN,'mae',NaN,'r2log',NaN,'qlike',NaN);
end
function p = pv(val); p = struct('h', val < 0.05, 'pValue', val); end
function nm = svParamNames(K)
    base = {'mu','phi','sigma_eta','rho','nu','beta_0','beta_1','v_h','v_r','v_omega'};
    cov  = arrayfun(@(k) sprintf('v_z_%d', k), 1:K, 'UniformOutput', false);
    nm   = [base, cov, {'w_h','b'}];
end
