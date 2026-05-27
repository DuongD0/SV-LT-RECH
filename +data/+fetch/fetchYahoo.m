function tbl = fetchYahoo(ticker, opts)
% fetchYahoo  Pull a Yahoo Finance daily series via the yfinance Python lib.
%
%   tbl = fetchYahoo(ticker)
%   tbl = fetchYahoo(ticker, opts)
%
%   ticker : Yahoo symbol, e.g. '^GSPC', '^N225', '^DJI', '^GDAXI',
%            '^FTSE', '^AORD', '^FCHI', '^BVSP', '^AEX', '^BFX', 'BTC-USD'.
%   opts   : struct
%              .start       'YYYY-MM-DD' default '2014-01-01'
%              .end         'YYYY-MM-DD' default today
%              .autoAdjust  logical default false  (keep raw OHLC)
%
%   Returns a table with columns:
%     date  (datetime)
%     open, high, low, close, adjClose, volume   (double)
%
%   Yahoo's direct v7 CSV endpoint now requires cookies / crumbs.
%   We pull through Python yfinance (the venv installs it; setup.sh
%   activates the venv).
%
%   Designed to be wrapped by cacheFetch.

    arguments
        ticker (1,:) char
        opts         struct = struct()
    end

    defaults.start      = '2014-01-01';
    defaults.end        = datestr(datetime('today'), 'yyyy-mm-dd');
    defaults.autoAdjust = false;
    opts = mergeStruct(defaults, opts);

    yf = py.importlib.import_module('yfinance');
    df = yf.download(ticker, ...
        pyargs('start',       opts.start, ...
               'end',         opts.end, ...
               'auto_adjust', opts.autoAdjust, ...
               'progress',    false, ...
               'group_by',    'column'));

    if df.empty
        error('fetchYahoo:empty', 'yfinance returned empty frame for %s', ticker);
    end

    %% Round-trip through a CSV — Python writes, MATLAB reads. Avoids
    %  Pandas MultiIndex / datetime conversion pain.
    tmpCsv  = [tempname, '.csv'];
    df.to_csv(tmpCsv);
    cleanup = onCleanup(@() delete(tmpCsv)); %#ok<NASGU>

    raw = readtable(tmpCsv, 'TextType', 'string', 'VariableNamingRule', 'preserve');

    headers = string(raw.Properties.VariableNames);
    dateIdx = find(strcmpi(headers, 'Date') | strcmpi(headers, 'Datetime'), 1);
    if isempty(dateIdx); dateIdx = 1; end

    dt = datetime(string(raw{:, dateIdx}), 'InputFormat', 'yyyy-MM-dd');

    function v = grab(name)
        ix = find(contains(headers, name, 'IgnoreCase', true), 1);
        if isempty(ix)
            v = nan(numel(dt), 1);
        else
            v = str2double(string(raw{:, ix}));
        end
    end

    open     = grab('Open');
    high     = grab('High');
    low      = grab('Low');
    closeP   = grab('Close');
    adjClose = grab('Adj');
    if all(isnan(adjClose)); adjClose = closeP; end
    volume   = grab('Volume');

    %% Drop header-residual rows where date didn't parse
    keep = ~isnat(dt) & ~all(isnan([open, high, low, closeP, volume]), 2);
    tbl  = table(dt(keep), open(keep), high(keep), low(keep), ...
                 closeP(keep), adjClose(keep), volume(keep), ...
                 'VariableNames', ...
                 {'date','open','high','low','close','adjClose','volume'});
    tbl  = sortrows(tbl, 'date');
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
