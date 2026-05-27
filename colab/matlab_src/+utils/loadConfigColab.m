function cfg = loadConfigColab(varargin)
% loadConfigColab  Colab-friendly JSON config loader.
%
%   cfg = loadConfigColab(file1, file2, ...)
%
%   Reads JSON configs via the built-in `jsondecode`. Mirrors the local
%   utils.loadConfig contract: later files override earlier files at the
%   leaf level (deep merge on nested structs).
%
%   Why JSON instead of YAML on Colab?
%     - readstruct's YAML support is not in every MATLAB release
%       (R2026a on the local box still lists only auto/json/xml).
%     - jsondecode is present in every supported release and has no
%       Python interop dependency.
%     - The Colab bundle ships JSON copies of every YAML config, so
%       this loader is a drop-in for the PyYAML path.

    arguments (Repeating)
        varargin (1,1) string
    end

    cfg = struct();
    for k = 1:numel(varargin)
        path = char(varargin{k});
        if ~isfile(path)
            error('loadConfigColab:fileNotFound', ...
                  'Config not found: %s', path);
        end
        next = readJsonAsStruct(path);
        cfg  = deepMerge(cfg, normalise(next));
    end
end


function s = readJsonAsStruct(path)
    fid = fopen(path, 'r');
    raw = char(fread(fid)');
    fclose(fid);
    s = jsondecode(raw);
end


function s = normalise(s)
    if ~isstruct(s); return; end
    fn = fieldnames(s);
    for i = 1:numel(fn)
        v = s.(fn{i});
        if isstruct(v)
            s.(fn{i}) = normalise(v);
        elseif isstring(v) && isscalar(v)
            s.(fn{i}) = char(v);
        end
    end

    % cell2mat(cfg.dgp.theta_true) expects a cell array. jsondecode
    % returns JSON numeric arrays as MATLAB column vectors; wrap so the
    % downstream interface matches the PyYAML path exactly.
    if isfield(s, 'theta_true') && isnumeric(s.theta_true)
        s.theta_true = num2cell(s.theta_true(:)');
    end
end


function a = deepMerge(a, b)
    fb = fieldnames(b);
    for k = 1:numel(fb)
        key = fb{k};
        if isfield(a, key) && isstruct(a.(key)) && isstruct(b.(key))
            a.(key) = deepMerge(a.(key), b.(key));
        else
            a.(key) = b.(key);
        end
    end
end
