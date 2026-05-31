# Standalone, single-file MATLAB models

Each `.m` file here is **fully self-contained**: drop it on the MATLAB path and run
it. No `+package` folders, no `addpath`, no other file from this repo is needed.
Every file inlines its own copy of the SMC/particle-filter engine (for the SV
models), the GARCH machinery, the evaluation metrics, and the ARMA mean-equation
gate. The mathematics is copied verbatim from the unit-tested `+package` codebase.

## Files

| File | Model | Backbone | Deep learning? | Inference |
|------|-------|----------|----------------|-----------|
| `run_comparison.m`   | **All 11 models + DM tests + Model Confidence Set** | — | — | — |
| `model_GARCH.m`      | GARCH-t / GJR-t            | GARCH | no  | MLE (Econometrics Tbx) |
| `model_GARCH_RECH.m` | GARCH-RECH (SRN/LSTM/GRU)  | GARCH | **yes** | MLE |
| `model_SV.m`         | Standard SV               | SV    | no  | SMC |
| `model_SVM.m`        | SV-in-Mean                | SV    | no  | SMC |
| `model_SVLT.m`       | SV + leverage + Student-t | SV    | no  | SMC |
| `model_SVLTRECH.m`   | SV-LT-RECH (SRN/LSTM/GRU) | SV    | **yes** | SMC |

`run_comparison.m` is the one to run for the headline result; the per-model files
are for inspecting or re-running a single model in isolation.

## How to run

```matlab
% Full comparison on the built-in synthetic demo (no data needed):
results = run_comparison;

% On your own returns y (T-by-1), optional covariates Z (T-by-K):
results = run_comparison(y, Z, splitIdx);

% A single model:
out = model_SVLTRECH(y, Z, splitIdx, 'lstm');   % SV deep learning (LSTM cell)
out = model_GARCH_RECH(y, Z, splitIdx, 'gru');  % GARCH deep learning (GRU cell)
out = model_SVM(y);                              % SV-in-mean, 80/20 split, no covariates
out = model_GARCH(y, [], 'gjr');                 % GJR-GARCH-t baseline

% Speed knobs (5th arg opts for the model files, 4th for run_comparison):
opts.smcN = 2000; opts.smcM = 300; opts.forecastJ = 200;   % SV accuracy
opts.grRestarts = 6;                                       % GARCH-RECH MLE restarts
opts.meanForce = 'arma';                                   % force ARMA mean (else 'auto'/'zero')
results = run_comparison(y, Z, splitIdx, opts);
```

Call any file with **no arguments** to run a built-in synthetic demo.

## Requirements per file

- **All files**: MATLAB R2026a (or recent) + Statistics and Machine Learning Toolbox.
- `model_GARCH.m` and the classic GARCH rows of `run_comparison.m`: **Econometrics
  Toolbox** (for `garch`/`gjr`/`estimate`). If absent, `run_comparison` skips those
  rows automatically.
- `model_GARCH_RECH.m`, `model_SV*.m`, `model_SVLTRECH.m`: **Optimization Toolbox**
  is used by the GARCH-RECH MLE (`fminunc`) but it falls back to base-MATLAB
  `fminsearch` if absent. The RNN cells are hand-coded — **no Deep Learning Toolbox
  required.**
- The ARMA mean-equation gate uses `lbqtest`/`arima` (Econometrics Toolbox). If that
  toolbox is missing, the gate falls back to zero-mean and the SV / GARCH-RECH models
  still run.

## Runtime

The SV models use Sequential Monte Carlo; runtime scales with `opts.smcN` ×
`opts.smcM` × series length. The synthetic demo (`run_comparison` with no args,
T = 400) runs in a few minutes; a full empirical run with `smcN = 2000` over six SV
models can take tens of minutes. The GARCH and GARCH-RECH models are fast (seconds).

## Verification note

These files are mechanical, verbatim flattenings of the `+package` source, which is
covered by the unit-test suite (`runtests('tests')` in the repo root). They were
**not executed in the environment that generated them** (no MATLAB license there).
Run `runtests('tests')` once in your environment to confirm the engine, then use
these files freely.
