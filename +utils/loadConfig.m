function cfg = loadConfig(varargin)
% loadConfig  Read YAML config(s) and deep-merge into a struct.
%
%   cfg = loadConfig(file1, file2, ...)
%
%   Later files override earlier files at the leaf level (deep merge on
%   nested structs). Typical usage:
%
%       cfg = utils.loadConfig('config/smc_defaults.yaml', ...
%                              'config/experiments/stage1.yaml');
%
%   Uses Python's PyYAML via MATLAB-Python interop (set up by setup.sh).

    arguments (Repeating)
        varargin (1,1) string
    end

    cfg = struct();
    for k = 1:numel(varargin)
        next = parseYaml(varargin{k});
        cfg  = deepMerge(cfg, next);
    end
end


function s = parseYaml(path)
    if ~isfile(path)
        error('loadConfig:fileNotFound', 'YAML not found: %s', path);
    end

    pyYaml = py.importlib.import_module('yaml');
    fid    = py.builtins.open(path, 'r');
    cleanup = onCleanup(@() fid.close());
    obj    = pyYaml.safe_load(fid);
    s      = py2matlab(obj);
end


function out = py2matlab(obj)
    %% Recursive conversion of Python objects to MATLAB equivalents.
    if isa(obj, 'py.dict')
        out  = struct();
        keys = cell(py.list(py.builtins.list(obj.keys())));
        for k = 1:numel(keys)
            key = char(keys{k});
            val = py2matlab(obj{keys{k}});
            out.(matlab.lang.makeValidName(key)) = val;
        end
    elseif isa(obj, 'py.list') || isa(obj, 'py.tuple')
        n   = length(obj);
        tmp = cell(1, n);
        for i = 1:n
            tmp{i} = py2matlab(obj{i});
        end
        out = tmp;
    elseif isa(obj, 'py.str')
        out = char(obj);
    elseif isa(obj, 'py.bool')
        out = logical(obj);
    elseif isa(obj, 'py.int') || isa(obj, 'py.float')
        out = double(obj);
    elseif isa(obj, 'py.NoneType')
        out = [];
    else
        out = obj;
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
