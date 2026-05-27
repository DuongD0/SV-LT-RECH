# SV-LT-RECH

MATLAB research codebase for **SV-LT-RECH** — Stochastic Volatility with Leverage, Student-t innovations, and an RNN-augmented log-variance state with exogenous covariates. The reference design is `PROPOSED_METHODOLOGY.md`.

Extension of:

- Nguyen, Nguyen & Tran (2024). *Deep learning enhanced volatility modeling with covariates* (RECH-X). *Finance Research Letters* 70, 106145.
- NEU Hanoi (2023). *Building Statistical Machine Learning Models for Volatility Forecasting*.

Both papers swap their GARCH backbone for a Stochastic Volatility backbone; everything else (Student-t innovations, leverage, RNN-augmented long-term volatility, exogenous covariates, Bayesian SMC inference) is preserved or strengthened.

## Layout

```
+models/         abstract Model base class + concrete model classes
+cells/          RNN cell forward passes (SRN / LSTM / GRU)
+inference/      +smc, +pf, +mcmc, +vb subpackages
+diagnostics/    posterior trace, R-hat, MCSE, sign relabeling
+data/           +fetch (pinned manifests) and +preprocess
+eval/           forecast metrics + DM test + MCS
+priors/         per-model log-prior densities
+utils/          numerical primitives, config loader, parpool helper
scripts/         reproducibility entry points (run_stage1..stage5)
notebooks/       MATLAB Live Scripts (.mlx) for narrative reports
config/          YAML configs (smc_defaults, priors, markets, experiments/)
tests/           MATLAB unittest framework
data/, results/  gitignored
docs/            data_sources, reproducibility_checklist, design_decisions
```

## Quick start

```bash
# One-time install (Python venv + conda env for R)
cd /home/d0/projects/finance_eng
python3 -m venv .venv
source .venv/bin/activate && pip install -r requirements.txt
deactivate
conda create -n sv-lt-rech-r -c conda-forge r-base r-stochvol -y

# Every terminal
source setup.sh
```

```matlab
cd('/home/d0/projects/finance_eng');
addpath(pwd);                            % +packages on the path

% Phase 1 sanity: plain SV via our SMC vs R stochvol
run('scripts/run_phase1_smc_sanity.m');

% Phase 3 simulation: 30 replicate identifiability sweep
run('scripts/run_stage1_simulation.m');
```

## Requirements

- **MATLAB R2026a** (this install — see `docs/reproducibility_checklist.md`).
  Toolboxes: Econometrics, Statistics and Machine Learning, Parallel Computing, Optimization.
- **Python 3.10+** in `.venv/` — `numpy`, `pandas`, `pyyaml`, `requests`,
  `yfinance`, `pandas-datareader`, `vnstock`, `jupyterlab`.
  Install: `python3 -m venv .venv && pip install -r requirements.txt`.
- **R 4.2+** in conda env `sv-lt-rech-r` — `stochvol >= 3.2.4` for the
  cross-engine SV calibration target.
  Install: `conda create -n sv-lt-rech-r -c conda-forge r-base r-stochvol -y`.

## Reproducibility

Every experiment is driven by a YAML config under `config/experiments/`. Random seeds and Monte Carlo SE are reported alongside every posterior. See `docs/reproducibility_checklist.md`.

## Status

Phase 1 + Phase 2 complete and verified — `runtests('tests')` reports **30/30 passing** in MATLAB R2026a. See `docs/PHASE_1_2_COMPLETE.md` for the full inventory and the Phase 3 specification.

## License

MIT (this repository). Reference papers retain their own licensing; `methodlogy_ref/` content is included under fair use for academic review.
