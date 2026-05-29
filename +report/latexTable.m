function tex = latexTable(T, opts)
% latexTable  Render a MATLAB table as a booktabs LaTeX tabular.
%
%   tex = report.latexTable(T, opts)
%
%   opts (all optional):
%     .lowerBetter   cellstr of numeric columns where the min is bolded
%     .higherBetter  cellstr of numeric columns where the max is bolded
%     .caption       table caption
%     .label         \label{...}
%     .precision     sig digits for numeric cells (default 3)
%     .file          if set, also writes the .tex to this path
%
%   NaN / empty numeric cells render as "--". Returns the LaTeX char.

    arguments
        T    table
        opts struct = struct()
    end
    defaults.lowerBetter  = {};
    defaults.higherBetter = {};
    defaults.caption      = '';
    defaults.label        = '';
    defaults.precision    = 3;
    defaults.file         = '';
    opts = mergeStruct(defaults, opts);

    vars = T.Properties.VariableNames;
    nCol = numel(vars);
    nRow = height(T);
    fmt  = sprintf('%%.%dg', opts.precision);

    bestRow = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for c = 1:nCol
        v = vars{c};
        if iscell(T.(v)) || ~isnumeric(T.(v)); continue; end
        col = T.(v);
        if ismember(v, opts.lowerBetter)
            [~, bestRow(v)] = min(col);
        elseif ismember(v, opts.higherBetter)
            [~, bestRow(v)] = max(col);
        end
    end

    L = strings(0, 1);
    L(end+1) = "\begin{table}[htbp]";
    L(end+1) = "\centering";
    if ~isempty(opts.caption); L(end+1) = "\caption{" + string(opts.caption) + "}"; end
    if ~isempty(opts.label);   L(end+1) = "\label{"  + string(opts.label)   + "}"; end
    L(end+1) = "\begin{tabular}{" + string(repmat('l', 1, nCol)) + "}";
    L(end+1) = "\toprule";
    L(end+1) = strjoin(escapeCells(vars), " & ") + " \\";
    L(end+1) = "\midrule";
    for r = 1:nRow
        cells = strings(1, nCol);
        for c = 1:nCol
            v = vars{c};
            cells(c) = fmtCell(T.(v), r, v, fmt, bestRow);
        end
        L(end+1) = strjoin(cells, " & ") + " \\"; %#ok<AGROW>
    end
    L(end+1) = "\bottomrule";
    L(end+1) = "\end{tabular}";
    L(end+1) = "\end{table}";

    tex = char(strjoin(L, newline));
    if ~isempty(opts.file)
        fid = fopen(opts.file, 'w'); fwrite(fid, tex); fclose(fid);
    end
end


function s = fmtCell(col, r, v, fmt, bestRow)
    if iscell(col)
        c = escapeCells(col(r));
        s = c(1);
        return;
    end
    if ~isnumeric(col)
        s = string(col(r)); return;
    end
    x = col(r);
    if isempty(x) || ~isfinite(x)
        s = "--"; return;
    end
    s = string(sprintf(fmt, x));
    if isKey(bestRow, v) && bestRow(v) == r
        s = "\textbf{" + s + "}";
    end
end


function out = escapeCells(c)
    out = strings(1, numel(c));
    for k = 1:numel(c)
        out(k) = replace(string(c{k}), "_", "\_");
    end
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
