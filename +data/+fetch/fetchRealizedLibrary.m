function tbl = fetchRealizedLibrary(entry, opts)
% fetchRealizedLibrary  Pull realised-volatility panel from a pinned URL.
%
%   tbl = fetchRealizedLibrary(entry)
%   tbl = fetchRealizedLibrary(entry, opts)
%
%   entry  : manifest entry struct with field `url` (and optional
%            `fallbackUrl`). For the Liu et al. (2025) RealRECH panel
%            this is the github.com/VBayesLab/RealRECH raw URL. The
%            Oxford-Man Realized Library was decommissioned 2022, so
%            this fetcher is the only path to RV.
%   opts   : struct
%              .timeout      webread timeout seconds (default 60)
%              .longFormat   true to reshape wide -> long (default true)
%
%   Returns a long-format table:
%     date    (datetime)
%     symbol  (string)
%     rv5     (double, realised variance from 5-min returns)
%
%   If the primary URL fails, falls back to `entry.fallbackUrl` if set.

    arguments
        entry  struct
        opts   struct = struct()
    end

    defaults.timeout    = 60;
    defaults.longFormat = true;
    opts = mergeStruct(defaults, opts);

    tmp     = [tempname, '.csv'];
    cleanup = onCleanup(@() delete(tmp));  %#ok<NASGU>

    urls = {entry.url};
    if isfield(entry, 'fallbackUrl') && ~isempty(entry.fallbackUrl)
        urls{end+1} = entry.fallbackUrl;
    end

    success = false;
    lastErr = '';
    for k = 1:numel(urls)
        try
            websave(tmp, urls{k}, weboptions('Timeout', opts.timeout));
            success = true;
            break
        catch ME
            lastErr = ME.message;
            continue
        end
    end
    if ~success
        error('fetchRealizedLibrary:downloadFailed', ...
              'All URLs failed for %s. Last error: %s', entry.name, lastErr);
    end

    raw = readtable(tmp, 'TextType', 'string', 'VariableNamingRule', 'preserve');

    headers = string(raw.Properties.VariableNames);
    dateIdx = find(contains(headers, 'date', 'IgnoreCase', true), 1);
    if isempty(dateIdx); dateIdx = 1; end

    dt = parseFlexibleDate(raw{:, dateIdx});
    raw{:, dateIdx} = string(dt, 'yyyy-MM-dd');

    if ~opts.longFormat
        tbl = raw;
        return
    end

    %% Reshape wide -> long (date, symbol, rv5)
    valueCols = setdiff(1:numel(headers), dateIdx);
    out = {};
    for k = valueCols
        sym = headers(k);
        if startsWith(sym, '.') || strlength(sym) == 0; continue; end
        rv5 = str2double(string(raw{:, k}));
        keep = ~isnan(rv5) & ~isnat(dt);
        if any(keep)
            out{end+1} = table(dt(keep), repmat(sym, sum(keep), 1), rv5(keep), ...
                'VariableNames', {'date', 'symbol', 'rv5'}); %#ok<AGROW>
        end
    end
    if isempty(out)
        error('fetchRealizedLibrary:noColumns', ...
              'No numeric RV columns found in %s', entry.name);
    end
    tbl = vertcat(out{:});
    tbl = sortrows(tbl, {'symbol', 'date'});
end


function dt = parseFlexibleDate(strs)
    s = string(strs);
    formats = {'yyyy-MM-dd', 'yyyyMMdd', 'dd/MM/yyyy', 'MM/dd/yyyy'};
    for k = 1:numel(formats)
        try
            dt = datetime(s, 'InputFormat', formats{k});
            return
        catch
            continue
        end
    end
    dt = datetime(s);
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
