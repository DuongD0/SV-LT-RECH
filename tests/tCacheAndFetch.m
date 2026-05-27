classdef tCacheAndFetch < matlab.unittest.TestCase
% tCacheAndFetch  Coverage for the cacheFetch manifest-aware cache.
%
%   Real fetchers hit the network — out of scope for unit tests. We
%   exercise cacheFetch with a synthetic in-memory fetcher and a
%   minimal in-memory manifest, so the cache + checksum + hit/miss
%   logic is verified without external dependencies.

    properties
        tmpRoot
    end

    methods (TestMethodSetup)
        function makeTmp(testCase)
            testCase.tmpRoot = tempname;
            mkdir(fullfile(testCase.tmpRoot, '+data', '+fetch'));
            mkdir(fullfile(testCase.tmpRoot, 'data', 'raw'));
        end
    end

    methods (TestMethodTeardown)
        function delTmp(testCase)
            if isfolder(testCase.tmpRoot)
                rmdir(testCase.tmpRoot, 's');
            end
        end
    end

    methods (Test)

        function freshThenCachedHit(testCase)
            manifest.fred = {struct('name', 'TEST', 'series', 'TEST', ...
                                    'snapshotDate', '2026-05-16', ...
                                    'sha256', 'pending')};
            payload = synthPayload();
            fetcher = @(e) payload;
            opts    = struct('root', testCase.tmpRoot, 'manifest', manifest);

            [t1, m1] = data.fetch.cacheFetch('fred', 'TEST', fetcher, opts);
            testCase.verifyEqual(m1.source, 'fresh');
            testCase.verifyEqual(height(t1), height(payload));

            [t2, m2] = data.fetch.cacheFetch('fred', 'TEST', fetcher, opts);
            testCase.verifyEqual(m2.source, 'cache');
            testCase.verifyEqual(height(t2), height(payload));
            testCase.verifyEqual(m1.sha256, m2.sha256);
        end

        function forceBypassesCache(testCase)
            manifest.fred = {struct('name', 'TEST', 'series', 'TEST', ...
                                    'snapshotDate', '2026-05-16', ...
                                    'sha256', 'pending')};
            payload = synthPayload();
            fetcher = @(e) payload;
            opts0 = struct('root', testCase.tmpRoot, 'manifest', manifest);

            [~, m1] = data.fetch.cacheFetch('fred', 'TEST', fetcher, opts0);
            testCase.verifyEqual(m1.source, 'fresh');

            opts1 = opts0; opts1.force = true;
            [~, m2] = data.fetch.cacheFetch('fred', 'TEST', fetcher, opts1);
            testCase.verifyEqual(m2.source, 'fresh');
        end

        function checksumMismatchAborts(testCase)
            manifest.fred = {struct('name', 'TEST', 'series', 'TEST', ...
                                    'snapshotDate', '2026-05-16', ...
                                    'sha256', repmat('a', 1, 64))};
            payload = synthPayload();
            fetcher = @(e) payload;
            opts    = struct('root', testCase.tmpRoot, 'manifest', manifest);

            testCase.verifyError( ...
                @() data.fetch.cacheFetch('fred', 'TEST', fetcher, opts), ...
                'cacheFetch:freshChecksumMismatch');
        end

        function unknownEntryFails(testCase)
            manifest.fred = {struct('name', 'OTHER', 'series', 'X', ...
                                    'snapshotDate', '2026-05-16', ...
                                    'sha256', 'pending')};
            opts = struct('root', testCase.tmpRoot, 'manifest', manifest);
            testCase.verifyError( ...
                @() data.fetch.cacheFetch('fred', 'MISSING', @(e) [], opts), ...
                'cacheFetch:unknownEntry');
        end
    end
end


function tbl = synthPayload()
    dates = string(datetime(2025,1,1) + caldays(0:9), 'yyyy-MM-dd').';
    val   = (1:10).';
    tbl   = table(dates, val, 'VariableNames', {'date', 'value'});
end
