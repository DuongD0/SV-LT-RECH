function out = modelConfidenceSet(losses, opts)
% modelConfidenceSet  Hansen-Lunde-Nason (2011) MCS via iterative elimination.
%
%   out = modelConfidenceSet(losses)
%   out = modelConfidenceSet(losses, opts)
%
%   losses : T_test-by-K matrix of per-period losses, one column per model.
%   opts   : struct
%              .alpha       confidence level for MCS (default 0.25 -> 75% set)
%              .B           bootstrap replicates (default 5000)
%              .blockLength stationary-bootstrap mean block length L
%                            (default round(sqrt(T)))
%              .statistic   'tMax' (default) or 'tRange' (HLN 2011 §3.1.1)
%              .seed        rng seed for bootstrap (default 20260516)
%
%   Outputs
%   -------
%     out.inSet            : K-by-1 logical, true for models in MCS
%     out.pValues          : K-by-1 running MCS p-value (monotone in
%                            elimination order)
%     out.eliminationOrder : K-by-1; 0 means never eliminated (in set)
%     out.statistic        : statistic used
%
%   References
%   ----------
%   Hansen, Lunde, Nason (2011). The model confidence set. Econometrica 79(2).
%   Politis, Romano (1994). The stationary bootstrap. JASA 89(428).

    arguments
        losses (:,:) double
        opts         struct = struct()
    end

    [T, K] = size(losses);
    assert(K >= 2, 'modelConfidenceSet:tooFewModels', ...
        'Need at least 2 models, got %d', K);

    defaults.alpha       = 0.25;
    defaults.B           = 5000;
    defaults.blockLength = max(2, round(sqrt(T)));
    defaults.statistic   = 'tMax';
    defaults.seed        = 20260516;
    opts = mergeStruct(defaults, opts);

    rngState = rng();
    cleanup  = onCleanup(@() rng(rngState));  %#ok<NASGU>
    rng(opts.seed, 'threefry');

    indices = stationaryBootstrapIndices(T, opts.B, opts.blockLength);

    active    = true(K, 1);
    pValues   = zeros(K, 1);
    elimOrder = zeros(K, 1);
    elimStep  = 0;

    while sum(active) > 1
        idx = find(active);
        Kj  = numel(idx);
        L   = losses(:, idx);
        meanL = mean(L, 1);
        D     = meanL.' - meanL;

        %% Bootstrap variance of pairwise diffs
        varD = zeros(Kj, Kj);
        for b = 1:opts.B
            Lb     = L(indices(:, b), :);
            meanLb = mean(Lb, 1);
            Db     = meanLb.' - meanLb;
            varD   = varD + (Db - D).^2;
        end
        varD = varD / opts.B;
        varD(varD < 1e-12) = 1e-12;
        sdD  = sqrt(varD);

        switch opts.statistic
            case 'tMax'
                t = max(D ./ sdD, [], 2);
                tStat = max(t);
            case 'tRange'
                t = (max(D, [], 2) - min(D, [], 2)) ./ max(sdD, [], 2);
                tStat = max(t);
            otherwise
                error('modelConfidenceSet:badStat', 'Unknown statistic');
        end

        tBoot = zeros(opts.B, 1);
        for b = 1:opts.B
            Lb      = L(indices(:, b), :);
            meanLb  = mean(Lb, 1);
            Db      = meanLb.' - meanLb;
            centred = Db - D;
            switch opts.statistic
                case 'tMax'
                    tBoot(b) = max(max(centred ./ sdD, [], 2));
                case 'tRange'
                    tBoot(b) = max((max(centred, [], 2) - min(centred, [], 2)) ./ max(sdD, [], 2));
            end
        end

        pVal = mean(tBoot >= tStat);
        pValues(idx) = max(pValues(idx), pVal);

        if pVal > opts.alpha
            break
        end

        [~, worst]            = max(t);
        worstAbs              = idx(worst);
        elimStep              = elimStep + 1;
        elimOrder(worstAbs)   = elimStep;
        active(worstAbs)      = false;
    end

    out.inSet            = active;
    out.pValues          = pValues;
    out.eliminationOrder = elimOrder;
    out.statistic        = opts.statistic;
end


function I = stationaryBootstrapIndices(T, B, L)
% Politis-Romano stationary bootstrap with geometric block lengths.
    p = 1 / L;
    I = zeros(T, B);
    for b = 1:B
        idx = randi(T);
        I(1, b) = idx;
        for t = 2:T
            if rand() < p
                idx = randi(T);
            else
                idx = idx + 1;
                if idx > T; idx = 1; end
            end
            I(t, b) = idx;
        end
    end
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
