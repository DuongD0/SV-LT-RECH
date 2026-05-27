# Design Decisions

Non-obvious choices, each with the alternative we rejected and the reason. Append-only.

## D1. MATLAB primary, R for baseline calibration only

We use MATLAB everywhere except for the `stochvol::svsample` baseline in `scripts/calibrate_against_stochvol.R`. Python was considered (better DL ecosystem, permissive licensing) but the reference papers and the closest published precedent code (VBayesLab) are MATLAB; mirroring that minimizes translation risk during replication.

## D2. Clean-room reimplementation, not fork

`VBayesLab/Stochastic-Volatility` and `VBayesLab/RealRECH` ship without LICENSE files. We re-read both and re-derived from the published papers and arXiv preprints; we do not copy code verbatim. `docs/design_decisions.md` records any algorithm we mirror.

## D3. `+models/` package with abstract base, NOT folder-per-model

Considered: one folder per model (`SVLTRECH/`, `SVLTRECHlstm/`, ...). Rejected: 80%+ of code duplicates across `*RECH*` variants. The abstract `Model` class collapses that to one cell-function override per variant.

## D4. SMC engine takes Model HANDLE objects, not function handles

MATLAB anonymous functions silently snapshot workspace state at creation time, which breaks under `parfor`. Handle classes propagate by reference and `save`/`load` cleanly with their hyperparameters.

## D5. Bootstrap PF in log-weight space from the first line

Linear-weight PFs underflow with Student-t innovations at ν ≈ 4 plus leverage plus an RNN log-variance perturbation. Log-sum-exp is mandatory, not optional. Auxiliary PF is wired in as a sibling for cases where bootstrap PF still degenerates.

## D6. Leverage via Cholesky coupling, mixture-of-normals as fallback

The Omori-Chib-Shephard-Nakajima (2007) mixture representation is more efficient but harder to debug. We start with Cholesky coupling of (εₜ, ηₜ₊₁) under the joint Gaussian assumption that holds for SV-LT (not SV-LT-t — for SV-LT-t the coupling is on the Gaussian part; the Student-t enters via a separate scale mixture). If particles starve we switch to the mixture.

## D7. Sign relabeling is a check, not a fix

SRN with ReLU has a sign symmetry that the priors β₀ ≥ 0, β₁ ≥ 0 are supposed to break — but β₀ = 0 is on the prior boundary. We post-process particles to enforce β₁ ≥ 0 and flag if the count of flipped particles is > 5% (suggests a label-switching mode that priors didn't catch).

## D8. Sequential phasing puts simulation BEFORE real data

The original plan had simulation study in Phase 4 (after benchmarks). The architect review flagged this: a simulator round-trip is the cheapest end-to-end test of the entire pipeline. If SV-LT-RECH can't recover known parameters on synthetic data, no amount of benchmark fitting matters. Simulation now runs in Phase 3.
