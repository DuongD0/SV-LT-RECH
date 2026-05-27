function [da, essAtNew] = adaptiveTemperature(logWcurrent, perParticleLogLik, aMax, targetEss)
% adaptiveTemperature  Bisect for the next anneal temperature increment.
%
%   [da, essAtNew] = adaptiveTemperature(logW, perParticleLogLik, aMax, targetEss)
%
%   Solves for da in (0, aMax] such that the conditional ESS at the new
%   tempered weights equals targetEss.
%
%   Conditional ESS at increment da:
%       logW_new = logW + da * perParticleLogLik
%       ess_new  = ESS(logW_new)
%
%   ESS is monotone non-increasing in da, so bisection on da works.
%   If ESS(da = aMax) already exceeds targetEss, return aMax — we can
%   take the full remaining step in one go.

    arguments
        logWcurrent       (:,1) double
        perParticleLogLik (:,1) double
        aMax              (1,1) double {mustBePositive}
        targetEss         (1,1) double {mustBePositive}
    end

    %% Quick check: full remaining step?
    essFull = essAt(logWcurrent, perParticleLogLik, aMax);
    if essFull >= targetEss
        da       = aMax;
        essAtNew = essFull;
        return
    end

    %% Bisect on da
    lo = 0;
    hi = aMax;
    for iter = 1:60
        mid    = 0.5 * (lo + hi);
        essMid = essAt(logWcurrent, perParticleLogLik, mid);
        if essMid > targetEss
            lo = mid;
        else
            hi = mid;
        end
        if hi - lo < 1e-10
            break
        end
    end
    da       = 0.5 * (lo + hi);
    essAtNew = essAt(logWcurrent, perParticleLogLik, da);
end


function e = essAt(logW, llh, da)
    e = inference.smc.ess(logW + da * llh);
end
