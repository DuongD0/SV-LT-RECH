# Phase 4 handoff archive — contents specification

Once the Colab Stage 1 study finishes, `package_results.m` produces:

```
phase4_handoff_<runId>_<timestamp>.tar.gz
```

Ship that one file back. When extracted, it contains:

```
.
├── MANIFEST.json                 ← run metadata + inline gate verdict
├── gate_summary.json             ← §4.5 acceptance criteria + verdict
├── recovery.csv                  ← per-(replicate, param) posterior recovery
├── mcse_logml.csv                ← per-replicate log Z + MCSE
├── sign_switch.csv               ← per-(replicate, seed) beta_1 sign diagnostic
├── posteriors/
│   ├── replicate1_seed1.mat      ← full posterior particles (N x K), per chain
│   ├── replicate1_seed2.mat
│   ├── ...
│   └── replicate30_seed3.mat
├── simulations/
│   ├── replicate1.mat            ← simulated (y, h_true, Z, theta_true)
│   ├── ...
│   └── replicate30.mat
└── figures/
    └── F11_replicate1_posterior.png
```

## File-by-file schema

### `MANIFEST.json`

```json
{
  "config_used":         "stage1_colab.yaml",
  "results_dir":         "results/stage1_colab",
  "archive_path":        "phase4_handoff_stage1_colab_20260520_034512.tar.gz",
  "packaged_at":         "20260520_034512",
  "gate_summary_inline": { ... copy of gate_summary.json ... }
}
```

### `gate_summary.json`

Quantitative §4.5 verdicts:

```json
{
  "stage1_complete_at":   "2026-05-20 03:45:12",
  "config_file":          "stage1_colab.yaml",
  "paramNames":           ["mu","phi","sigma_eta","rho","nu","beta_0","beta_1",
                           "v_h","v_r","v_omega","w_h","b"],
  "coverage95_per_param": [0.93, 0.97, 0.90, 0.87, 0.93, 0.97, 0.93, 0.90,
                           0.93, 0.93, 0.93, 0.97],
  "gates": {
    "coverage_95_min": {"value": 0.87, "target": ">= 0.9", "pass": false},
    "rhat_max":        {"value": 1.08, "target": "<= 1.1", "pass": true},
    "mcse_logz_max":   {"value": 0.42, "target": "<= 0.5", "pass": true},
    "sign_flip_max":   {"value": 0.0,  "target": "< 0.05", "pass": true}
  },
  "overall":    "FAIL",
  "next_step":  "..."
}
```

### `recovery.csv` (R*K rows)

| Column      | Type   | Meaning |
|-------------|--------|---------|
| replicate   | int    | 1..nReplicates |
| seed        | NaN    | unused (kept for column-shape compat with prior tables) |
| param       | string | parameter name from `paramNames` |
| true        | float  | ground-truth theta value |
| postMean    | float  | pooled posterior mean across seeds |
| postStd     | float  | pooled posterior std across seeds |
| withinCI95  | bool   | \|postMean - true\| <= 2 * postStd |
| rHat        | float  | split-R-hat across seeds for that parameter |

### `mcse_logml.csv` (R rows)

| Column     | Type  | Meaning |
|------------|-------|---------|
| logZ_mean  | float | mean log marginal likelihood across seeds |
| logZ_mcse  | float | Monte Carlo SE = std/sqrt(S) |
| replicate  | int   | 1..nReplicates |

### `sign_switch.csv` (R*S rows)

| Column      | Type  | Meaning |
|-------------|-------|---------|
| replicate   | int   | 1..nReplicates |
| seed        | int   | 1..nSeeds |
| fracFlipped | float | fraction of particles with beta_1 < 0 (should be 0) |

### `posteriors/replicate{R}_seed{S}.mat`

MATLAB struct, fields:

| Field            | Type       | Shape  | Meaning |
|------------------|------------|--------|---------|
| `theta`          | double     | N x K  | final posterior particles |
| `logLik`         | double     | N x 1  | log p(y \| theta_i) per particle |
| `logMarginalLik` | double     | scalar | log Z for this chain |
| `schedule`       | double     | step x 1 | anneal temperatures |
| `essTrace`       | double     | step x 1 | ESS at every step |
| `acceptTrace`    | double     | step x 1 | RWM acceptance per step |
| `paramNames`     | cellstr    | 1 x K  | parameter labels |

**Phase 4 uses these directly** as warm-start initial particles for
the LSTM / GRU / RealRECH variants — much faster SMC convergence than
sampling from prior.

### `simulations/replicate{R}.mat`

| Field      | Type   | Shape  | Meaning |
|------------|--------|--------|---------|
| `y`        | double | T x 1  | simulated return series |
| `hTrue`    | double | T x 1  | ground-truth latent log-variance |
| `Z`        | double | T x K_cov | covariate matrix (empty when K_cov=0) |
| `thetaTrue`| double | 1 x K  | ground-truth theta |

These let Phase 4 re-run any specific replicate against a NEW model
variant without redoing the (deterministic but expensive) simulator.

## What Phase 4 will do with this

1. Read `gate_summary.json` → confirm SVLTRECH-SRN-Cholesky baseline
   meets §4.5; if not, halt and fix before proceeding.
2. Read `recovery.csv` + `mcse_logml.csv` → extract baseline numbers
   for the Table T2 / T3 / T4 entries in the eventual paper.
3. For each Phase 4 variant (SVLTRECH-LSTM, SVLTRECH-GRU, RSV-LT-RECH):
   - Read one of the `simulations/replicate*.mat` files for the
     simulated `(y, theta_true)` pair.
   - Optionally warm-start SMC from the posterior in
     `posteriors/replicate*_seed1.mat`, padding the new parameters
     with prior draws.
   - Run the new SMC and compare log Z and recovery to the SRN
     baseline.

## Reproducibility

Every step is deterministic in the `baseSeed` from
`stage1_colab.yaml`. If you re-run from scratch (delete CSVs +
posteriors directory) on a fresh Colab VM, you'll get bit-identical
results (modulo MATLAB version differences in random-number sequences).

The Colab `runStage1Colab` runner also reads the existing CSV when
present and resumes from the next replicate, so multi-session runs
produce the same output as a single-session run.
