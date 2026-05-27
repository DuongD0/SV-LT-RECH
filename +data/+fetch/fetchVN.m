function tbl = fetchVN(symbol, opts)
% fetchVN  Pull a Vietnam stock-index daily series via the vnstock Python lib.
%
%   tbl = fetchVN(symbol)
%   tbl = fetchVN(symbol, opts)
%
%   symbol : 'VNINDEX' (HOSE composite), 'HNXINDEX' (HNX composite),
%            'UPCOMINDEX', or any HOSE ticker (e.g. 'VIC', 'FPT').
%   opts   : struct
%              .start    'YYYY-MM-DD' default '2016-01-04'
%              .end      'YYYY-MM-DD' default today
%              .source   vnstock data source ('VCI' default, 'TCBS', 'MSN')
%
%   Returns a table with columns:
%     date, open, high, low, close, volume
%
%   investing.com no longer permits programmatic access; vnstock pulls
%   from VCI / TCBS / Vietnam exchange APIs. setup.sh activates the
%   project venv so MATLAB's pyenv resolves to it.

    arguments
        symbol (1,:) char
        opts         struct = struct()
    end

    defaults.start  = '2016-01-04';
    defaults.end    = datestr(datetime('today'), 'yyyy-mm-dd');
    defaults.source = 'VCI';
    opts = mergeStruct(defaults, opts);

    vn  = py.importlib.import_module('vnstock');
    obj = vn.Vnstock().stock(pyargs('symbol', symbol, 'source', opts.source));
    df  = obj.quote.history(pyargs('start', opts.start, 'end', opts.end));

    if df.empty
        error('fetchVN:empty', 'vnstock returned empty frame for %s', symbol);
    end

    %% Round-trip via CSV (same pattern as fetchYahoo)
    tmpCsv  = [tempname, '.csv'];
    df.to_csv(pyargs('path_or_buf', tmpCsv, 'index', false));
    cleanup = onCleanup(@() delete(tmpCsv));  %#ok<NASGU>

    raw     = readtable(tmpCsv, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    headers = string(raw.Properties.VariableNames);

    function v = grab(name)
        ix = find(strcmpi(headers, name), 1);
        if isempty(ix); ix = find(contains(headers, name, 'IgnoreCase', true), 1); end
        if isempty(ix)
            v = nan(height(raw), 1);
        else
            v = str2double(string(raw{:, ix}));
        end
    end

    dateIdx = find(strcmpi(headers, 'time') | strcmpi(headers, 'date'), 1);
    if isempty(dateIdx); dateIdx = 1; end
    dt = datetime(string(raw{:, dateIdx}), 'InputFormat', 'yyyy-MM-dd');

    open   = grab('open');
    high   = grab('high');
    low    = grab('low');
    closeP = grab('close');
    volume = grab('volume');

    keep = ~isnat(dt) & ~all(isnan([open, high, low, closeP, volume]), 2);
    tbl  = table(dt(keep), open(keep), high(keep), low(keep), ...
                 closeP(keep), volume(keep), ...
                 'VariableNames', {'date','open','high','low','close','volume'});
    tbl  = sortrows(tbl, 'date');
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
