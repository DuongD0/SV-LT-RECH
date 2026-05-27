function ds = loadStage4(cfg)
% loadStage4  Fetch + align VN-Index and four covariates for Stage 4.
%
%   ds = data.loadStage4(cfg)
%
%   cfg.data fields: .start, .end, .trainEnd, .covariates (cellstr subset of
%   {'oil','usdvnd','btc','gold'}). Requires network + the project venv
%   (run `source setup.sh`, and pyenv pointed at .venv).
%
%   Returns struct `ds`:
%     .dates    n-by-1 datetime (trading days with a valid VN-Index close)
%     .y        n-by-1 returns, 100*log(close_t/close_{t-1})  (NaN-dropped)
%     .Z        n-by-K covariate matrix, RAW (NOT yet standardised)
%     .covNames 1-by-K cellstr
%     .splitIdx scalar = cfg.data.trainEnd (clamped to < n)
%
%   Standardisation is intentionally deferred to the caller so it can apply
%   train-only statistics after the chronological split (no leakage).

    arguments
        cfg (1,1) struct
    end

    optsP = struct('start', cfg.data.start, 'end', cfg.data.end);

    %% VN-Index close -> 100*log returns
    vnTbl = data.fetch.fetchVN('VNINDEX', optsP);
    px    = vnTbl.close;
    ret   = [NaN; 100 * log(px(2:end) ./ px(1:end-1))];
    base  = table(vnTbl.date, ret, 'VariableNames', {'date','y'});

    %% Covariates -> daily levels, joined on date
    src = struct( ...
        'oil',    'DCOILWTICO', ...        % WTI crude, USD/bbl (FRED)
        'gold',   'GOLDAMGBD228NLBM', ...  % London gold fixing (FRED)
        'usdvnd', 'DEXVNUS');              % USD/VND (FRED; verify availability)
    covNames = cfg.data.covariates;
    for c = 1:numel(covNames)
        name = covNames{c};
        switch name
            case {'oil','gold','usdvnd'}
                ft  = data.fetch.fetchFRED(src.(name), optsP);
                col = table(ft.date, ft.value, 'VariableNames', {'date', name});
            case 'btc'
                yt  = data.fetch.fetchYahoo('BTC-USD', optsP);
                col = table(yt.date, yt.close, 'VariableNames', {'date', name});
            otherwise
                error('loadStage4:unknownCovariate', 'Unknown covariate %s', name);
        end
        base = outerjoin(base, col, 'Keys', 'date', 'MergeKeys', true, 'Type', 'left');
    end

    %% Forward-fill covariates across non-trading gaps, then drop missing target
    base = sortrows(base, 'date');
    for c = 1:numel(covNames)
        base.(covNames{c}) = fillmissing(base.(covNames{c}), 'previous');
    end
    base = base(~isnan(base.y), :);
    Zraw = base{:, covNames};
    good = all(~isnan(Zraw), 2);          % drop leading rows with no prior to fill
    base = base(good, :);

    n           = height(base);
    ds.dates    = base.date;
    ds.y        = base.y;
    ds.Z        = base{:, covNames};
    ds.covNames = covNames;
    ds.splitIdx = min(cfg.data.trainEnd, n - 1);
end
