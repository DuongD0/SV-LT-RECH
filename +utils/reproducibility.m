function rngState = reproducibility(seed)
% reproducibility  Deterministic RNG setup for one run.
%
%   rngState = reproducibility(seed)
%
%   Resets the global stream to a `threefry` generator (substream-safe
%   for parfor) seeded with `seed`. Returns the previous state so callers
%   can restore it (useful inside tests).
%
%   Convention: experiment scripts pass `seed = baseSeed + replicate`.
%   See docs/reproducibility_checklist.md.

    arguments
        seed (1,1) {mustBeNonnegative, mustBeInteger}
    end

    rngState = rng();
    rng(seed, 'threefry');
end
