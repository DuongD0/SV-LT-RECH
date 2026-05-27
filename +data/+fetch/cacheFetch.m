function [tbl, meta] = cacheFetch(kind, name, fetcher, opts)
% cacheFetch  Manifest-aware filesystem cache for external data pulls.
%
%   [tbl, meta] = cacheFetch(kind, name, fetcher)
%   [tbl, meta] = cacheFetch(kind, name, fetcher, opts)
%
%   Inputs
%   ------
%     kind     : top-level manifest section ('yahoo', 'fred',
%                'realizedLibrary', 'investing', 'vietnam').
%     name     : entry name within the section (e.g. 'SPX', 'VIX').
%     fetcher  : function handle to call on cache miss. Signature:
%                  tbl = fetcher(entry)
%                where `entry` is the manifest struct (so the fetcher has
%                access to ticker, url, dates, etc.).
%     opts     : struct
%                  .root        repo root (default = pwd)
%                  .force       true to bypass cache (default false)
%                  .manifest    cached manifest struct (avoid re-parsing)
%                  .updateSha   true to overwrite stored sha256 stamp
%                               when manifest entry says 'pending'
%                               (default true)
%
%   Outputs
%   -------
%     tbl   : MATLAB table loaded from cache or freshly fetched.
%     meta  : struct with .source ('cache'|'fresh'), .path, .sha256.
%
%   Cache layout
%   ------------
%     data/raw/<kind>/<name>.csv      payload
%     data/raw/<kind>/<name>.sha256   sha256 stamp (text)
%
%   The manifest sha256 is authoritative: if it is set to a value other
%   than 'pending', cached and fresh content MUST match it byte-for-byte
%   or we abort. When 'pending', the first successful pull writes the
%   actual sha256 into the stamp file (the YAML is left for the user to
%   review and commit).

    arguments
        kind    (1,:) char
        name    (1,:) char
        fetcher (1,1) function_handle
        opts          struct = struct()
    end

    defaults.root      = pwd;
    defaults.force     = false;
    defaults.manifest  = [];
    defaults.updateSha = true;
    opts = mergeStruct(defaults, opts);

    %% Resolve manifest entry
    if isempty(opts.manifest)
        manifestPath = fullfile(opts.root, '+data', '+fetch', 'manifest.yaml');
        manifest     = utils.loadConfig(manifestPath);
    else
        manifest = opts.manifest;
    end
    entry = findEntry(manifest, kind, name);

    %% Resolve cache paths
    cacheDir = fullfile(opts.root, 'data', 'raw', kind);
    if ~isfolder(cacheDir); mkdir(cacheDir); end
    csvPath  = fullfile(cacheDir, [name, '.csv']);
    shaPath  = fullfile(cacheDir, [name, '.sha256']);

    expectedSha = lower(strtrim(entry.sha256));
    pinned      = ~strcmp(expectedSha, 'pending');

    %% Cache hit?
    if ~opts.force && isfile(csvPath)
        actualSha = sha256File(csvPath);
        if pinned && ~strcmp(actualSha, expectedSha)
            error('cacheFetch:checksumMismatch', ...
                'Cached %s/%s sha256 %s != manifest %s', kind, name, ...
                actualSha, expectedSha);
        end
        tbl  = readtable(csvPath, 'TextType', 'string');
        meta = struct('source', 'cache', 'path', csvPath, 'sha256', actualSha);
        return
    end

    %% Cache miss -> fetch fresh
    tbl       = fetcher(entry);
    writetable(tbl, csvPath);
    actualSha = sha256File(csvPath);

    if pinned && ~strcmp(actualSha, expectedSha)
        error('cacheFetch:freshChecksumMismatch', ...
            'Fresh %s/%s sha256 %s != manifest %s. Remote source changed?', ...
            kind, name, actualSha, expectedSha);
    end

    if opts.updateSha
        fid = fopen(shaPath, 'w');
        fprintf(fid, '%s\n', actualSha);
        fclose(fid);
    end

    meta = struct('source', 'fresh', 'path', csvPath, 'sha256', actualSha);
end


function entry = findEntry(manifest, kind, name)
    if ~isfield(manifest, kind)
        error('cacheFetch:unknownKind', 'No manifest section "%s"', kind);
    end
    section = manifest.(kind);
    if iscell(section)
        for k = 1:numel(section)
            if isfield(section{k}, 'name') && strcmp(section{k}.name, name)
                entry = section{k};
                return
            end
        end
    end
    error('cacheFetch:unknownEntry', 'No entry "%s" in section %s', name, kind);
end


function s = sha256File(path)
    raw    = fileread(path);
    bytes  = unicode2native(raw, 'UTF-8');
    md     = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    digest = typecast(md.digest(), 'uint8');
    s      = lower(reshape(dec2hex(digest, 2).', 1, []));
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
