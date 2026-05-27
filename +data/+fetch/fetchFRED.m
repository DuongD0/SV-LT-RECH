function tbl = fetchFRED(seriesId, opts)
% fetchFRED  Pull a FRED series via the public fredgraph.csv endpoint.
%
%   tbl = fetchFRED(seriesId)
%   tbl = fetchFRED(seriesId, opts)
%
%   seriesId : FRED series mnemonic, e.g. 'VIXCLS', 'DCOILWTICO',
%              'GOLDAMGBD228NLBM', 'DEXVNUS', 'INDPRO', 'UNRATE',
%              'CPIAUCSL'.
%   opts     : struct
%                .start  earliest date as 'YYYY-MM-DD' (default '2010-01-01')
%                .end    latest   date as 'YYYY-MM-DD' (default today)
%
%   Returns a table with columns:
%     date  (datetime)
%     value (double; NaN where FRED reports '.')
%
%   Designed to be wrapped by cacheFetch:
%
%       fetcher = @(entry) data.fetch.fetchFRED(entry.series, ...
%           struct('start', entry.start, 'end', entry.end));
%       [tbl, meta] = data.fetch.cacheFetch('fred', 'VIX', fetcher);

    arguments
        seriesId (1,:) char
        opts           struct = struct()
    end

    defaults.start = '2010-01-01';
    defaults.end   = datestr(datetime('today'), 'yyyy-mm-dd');
    opts = mergeStruct(defaults, opts);

    url = sprintf(['https://fred.stlouisfed.org/graph/fredgraph.csv', ...
                   '?id=%s&cosd=%s&coed=%s'], ...
                  seriesId, opts.start, opts.end);

    raw = webread(url, weboptions('Timeout', 60));

    %% Parse: header row + data rows
    rows = splitlines(strtrim(string(raw)));
    if numel(rows) < 2
        error('fetchFRED:empty', 'FRED returned no rows for %s', seriesId);
    end
    headers = split(rows(1), ',');
    payload = rows(2:end);

    dateColMask = strcmpi(headers, 'observation_date') | strcmpi(headers, 'DATE');
    if ~any(dateColMask)
        error('fetchFRED:badSchema', ...
              'FRED CSV missing date column. Headers: %s', strjoin(headers, ','));
    end

    parts = arrayfun(@(s) split(s, ',').', payload, 'UniformOutput', false);
    parts = vertcat(parts{:});

    dateIdx  = find(dateColMask, 1);
    dateStr  = parts(:, dateIdx);
    valueStr = parts(:, ~dateColMask);

    dt    = datetime(dateStr, 'InputFormat', 'yyyy-MM-dd');
    valueNum = str2double(valueStr);

    tbl = table(dt, valueNum, 'VariableNames', {'date', 'value'});
    tbl = sortrows(tbl, 'date');
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
