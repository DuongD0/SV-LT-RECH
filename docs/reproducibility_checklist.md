# Reproducibility Checklist

A run is reproducible when the seed, the config, and the data manifest collectively determine every number reported.

## Environment

### Verified on this host (2026-05-16)

| Interpreter | Version | Location |
|---|---|---|
| Python | 3.13.12 | `.venv/` (venv) |
| R | 4.5.3 | conda env `sv-lt-rech-r` |
| stochvol | 3.2.9 | conda env `sv-lt-rech-r` |
| MATLAB | R2026a + Econometrics + Stats/ML + Parallel + Optimization | `$HOME/MATLAB/R2026a/` (via mpm) |

`source setup.sh` activates the Python venv + the conda R env in one step
and exports `RSCRIPT_BIN` to the conda-managed Rscript binary.

### Install commands

```bash
# Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# R + stochvol (no sudo needed)
conda create -n sv-lt-rech-r -c conda-forge r-base r-stochvol -y
```

### MATLAB on WSL2 (R2026a via mpm)

Installed user-locally (no sudo) with MathWorks Package Manager:

```bash
# One-time
wget https://www.mathworks.com/mpm/glnxa64/mpm -O /tmp/mpm
chmod +x /tmp/mpm
mkdir -p $HOME/MATLAB/R2026a
/tmp/mpm install \
    --release=R2026a \
    --destination=$HOME/MATLAB/R2026a \
    --products MATLAB Econometrics_Toolbox \
               Statistics_and_Machine_Learning_Toolbox \
               Parallel_Computing_Toolbox \
               Optimization_Toolbox
```

`setup.sh` adds `$HOME/MATLAB/R2026a/bin` to `PATH` once present.

**First-run licence activation**: launch `matlab -nodesktop` once and
either sign in to your MathWorks account or drop a `license.lic` into
`$HOME/MATLAB/R2026a/licenses/`. Headless activation is documented at
<https://www.mathworks.com/help/install/ug/activate-a-license.html>.

### Required MATLAB Toolboxes

- Econometrics
- Statistics and Machine Learning
- Parallel Computing

## Seeds

- Every script calls `utils.reproducibility(seed)` first thing.
- `seed` defaults to 20260516 (today's date as YYYYMMDD).
- All `parfor` worker streams are derived deterministically from the master seed.
- Independent SMC runs use seeds `seed`, `seed+1`, ..., `seed+9` (ten replicates per market×model — matches RECH-X paper convention).

## Configs

- One YAML per experiment under `config/experiments/`.
- `loadConfig` resolves YAML → struct, validates required fields, fills defaults from `config/smc_defaults.yaml`.

## Data manifest

- `+data/+fetch/manifest.yaml` pins every external URL, an expected SHA-256, and a "snapshot date".
- `cacheFetch` aborts if checksum mismatches — never silently re-downloads or substitutes.

## Outputs

- Posteriors saved to `results/posteriors/<experiment>/<market>/<model>/seed_<n>.mat`.
- Forecasts to `results/forecasts/.../seed_<n>.csv`.
- Aggregated scores to `results/scores/<experiment>/<market>.parquet`.
- Tables T1–T7 / figures F1–F4 regenerated from these via `scripts/make_paper_tables.m`.

## Convergence

- Report Monte Carlo SE of log marginal likelihood across 10 independent SMC seeds — should be < 0.5 nats.
- Report R̂ across the 10 seeds for every parameter (`+diagnostics/rHat.m`) — flag if > 1.05.
- Report fraction of replicates where the 95% credible interval contains the true value in the §11.1 simulation study.
