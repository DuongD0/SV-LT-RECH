# Run Guide

A step-by-step guide to running this project's volatility-model comparison.
Two ways are provided: the **standalone single files** (easiest) and the full
**`+package` codebase with its test suite** (most complete).

---

## 1. What this compares

The project benchmarks the effectiveness of **deep-learning-augmented** volatility
models against classical ones, symmetrically across two backbones:

|                  | no deep learning              | + deep learning (SRN / LSTM / GRU)     |
|------------------|-------------------------------|-----------------------------------------|
| **GARCH** backbone | GARCH-t, GJR-t              | **GARCH-RECH**-{SRN,LSTM,GRU}          |
| **SV** backbone    | SV, **SVM** (in-mean), SVLT | **SV-LT-RECH**-{SRN,LSTM,GRU} (flagship) |

- **Standard SV** and **SV-in-Mean (SVM)** are both included (SVM puts the latent
  volatility into the *mean* equation — the risk-return trade-off).
- **GARCH-RECH** is the original Nguyen-Tran-Kohn (2022) RECH model and is the GARCH
  counterpart of the SV deep-learning models, so the DL effect is comparable on both
  backbones.
- Every model is fit on **ARMA mean-equation residuals** (an automatic Ljung-Box gate
  decides zero-mean vs ARMA(p,q)); models are ranked by **QLIKE**, **Diebold-Mariano**
  tests, and the **Model Confidence Set**.

---

## 2. Requirements

- **MATLAB R2026a** (or a recent release).
- Toolboxes:
  - **Statistics and Machine Learning** — required by everything.
  - **Econometrics** — classic GARCH-t/GJR-t and the ARMA mean-equation gate.
  - **Optimization** — GARCH-RECH MLE (`fminunc`); auto-falls back to `fminsearch`.
  - **Parallel Computing** — optional (speeds up the full simulation study only).
- **No Deep Learning Toolbox needed** — the RNN cells (SRN/LSTM/GRU) are hand-coded.

Check what you have:

```matlab
ver                              % full list
license('test','econometrics_toolbox')   % 1 = available
assert(~isempty(ver('econ')),  'Need Econometrics Toolbox')
assert(~isempty(ver('stats')), 'Need Statistics & ML Toolbox')
```

### Do I need to install any toolboxes?

**No.** The bundled `MATLAB_R2026a/` tree already contains MATLAB + every toolbox this
project uses, and `mpm list --matlabroot=./MATLAB_R2026a` confirms 5 products: MATLAB,
Statistics & ML, Econometrics, Optimization, Parallel Computing. Deep Learning Toolbox
is **not** required (the RNN cells are hand-coded). What matters is whether your
**license entitles** these — a toolbox is usable only when it is *both installed AND
licensed*. After activating, verify:

```matlab
ver
license('test','statistics_toolbox')      % 1 = OK (hard requirement, no fallback)
license('test','econometrics_toolbox')    % 1 = classic GARCH baselines + ARMA gate
license('test','optimization_toolbox')    % 1 = best GARCH-RECH fitting
```

The code **degrades gracefully**: without Econometrics it skips the classic GARCH rows
and falls back to zero-mean; without Optimization, GARCH-RECH uses base-MATLAB
`fminsearch`. Only **Statistics & ML is a hard requirement** (no fallback).

### Activating the bundled `MATLAB_R2026a/` install

The ~9.4 GB tree was downloaded via the MathWorks Package Manager (`mpm`). **It is
installed software with no license attached** (`MATLAB_R2026a/licenses/` is absent —
normal pre-activation). `mpm` installs files but never a license, so activation is a
separate step. Pick the method matching your license:

```bash
# (a) Individual / Named-User / Campus / Trial — online sign-in (simplest):
./MATLAB_R2026a/bin/matlab                          # first launch opens the sign-in dialog
#   sign in with the MathWorks Account holding your license -> a license file is written
#   into MATLAB_R2026a/licenses/. Or run the client directly:
./MATLAB_R2026a/bin/glnxa64/MathWorksProductAuthorizer.sh

# (c) Network License Manager (university/company server):
export MLM_LICENSE_FILE=27000@your-license-server   # port@host, then launch matlab
#   (or place a two-line SERVER/USE_SERVER network.lic in MATLAB_R2026a/licenses/ —
#    use ONE or the other; if you set MLM_LICENSE_FILE, remove licenses/network.lic)
```

- **(b) Offline / Designated-Computer:** on an internet machine, fetch a *File
  Installation Key* + a license file (bound to this machine's MAC Host ID) from
  <https://www.mathworks.com/licensecenter>, copy `license.lic` into
  `MATLAB_R2026a/licenses/`, then activate via `MathWorksProductAuthorizer.sh` →
  "Activate without an Internet connection".
- **Headless servers (no display):** the sign-in dialog needs a GUI/internet. For
  headless/CI use the separate `matlab-batch` tool with a Batch Licensing Token
  (`MLM_LICENSE_TOKEN`) — not bundled here — or a license file / network license.

If you already have a licensed MATLAB on your machine, **just use that** and ignore the
`MATLAB_R2026a/` folder (it is git-ignored and does not travel with the repo); only the
toolbox entitlements in the table above matter.

### Troubleshooting: MATLAB segfaults at startup (too-new / unsupported Linux, WSL2)

R2026a is validated only on **Ubuntu 22.04 / 24.04, Debian 12 / 13, RHEL 8 / 9,
SLES/SLED 15** (glibc ≥ 2.28); **WSL2 is officially unsupported**. On a newer OS
(e.g. Ubuntu 26.04 / glibc 2.43) the **GUI desktop segfaults at startup** — the crash
lands in `libmwsettingscore` / `agent::spf::frameworksetup` on an early worker thread.
Root cause: glibc 2.35+ restartable sequences (`rseq`) clash with MATLAB's bundled
`tcmalloc` allocator (MathWorks bug 2632298). The **compute core is unaffected** —
`matlab -batch` runs fine; only the desktop crashes.

Fixes, in order:

```bash
# 1) Disable glibc rseq (verified to stop the segfault). Use the wrapper:
./run_matlab.sh -batch "addpath(pwd); runtests('tests')"     # headless
./run_matlab.sh                                              # GUI
#    (equivalently: export GLIBC_TUNABLES=glibc.pthread.rseq=0)

# 2) If the GUI then complains 'Qt platform plugin "xcb" ... failed', install the
#    X libraries mpm does not install:
sudo apt-get update && sudo apt-get install -y \
  libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libgtk-3-0t64 libxt6t64 libxss1

# 3) Best long-term fix: run on a SUPPORTED OS (Ubuntu 24.04/22.04), or use native
#    Windows/macOS MATLAB. For this project the GUI is optional — `-batch` is enough.
```

This crash is specific to a too-new Linux/WSL2 host; on a supported MATLAB install
(which most graders have) it does not occur.

---

## 3. Easiest path — the standalone single files

No setup. Each file in `standalone/` is fully self-contained.

```matlab
cd standalone

results = run_comparison;          % full 11-model comparison on a built-in demo
% ... or on your data:
results = run_comparison(y, Z, splitIdx);

% Individual models:
model_SVLTRECH(y, Z, splitIdx, 'lstm');   % SV deep learning
model_GARCH_RECH(y, Z, splitIdx, 'srn');  % GARCH deep learning
model_SVM(y);                             % SV-in-mean
model_GARCH(y, [], 'gjr');                % GJR-GARCH-t
```

`run_comparison` prints a scores table, the Model Confidence Set, and Diebold-Mariano
tests of every model against the flagship. See `standalone/README.md` for the speed
knobs (`opts.smcN`, `opts.smcM`, `opts.forecastJ`, `opts.grRestarts`, `opts.meanForce`).

---

## 4. Full path — the `+package` codebase + tests

From the repository root:

```matlab
cd /path/to/SV-LT-RECH
addpath(pwd);                      % puts the +packages on the path

% (1) Run the unit-test suite — this is the authoritative correctness check:
results = runtests('tests');
disp(table(results));              % expect all passing

% (2) Run the full Stage-4 comparison on a synthetic demo (no network/data needed):
run('scripts/run_stage4_demo.m');  % writes tables + figures to results/stage4_demo/

% (3) Live Vietnam application (needs the data fetchers / network; see README):
%     run('scripts/run_stage4_vietnam.m');
```

The new models added in this round each carry their own tests:
`tSvmRoundTrip` (SV-in-mean), `tGarchRECH` (GARCH deep learning), and the expanded
`tStage4Models` / `tStage4Smoke` (the 11-model registry + ARMA wiring).

---

## 5. What to look for in the output

- **QLIKE** (primary, lower = better) and **PPS** (predictive score) on the test window.
- **Model Confidence Set (75%)** — the set of models statistically indistinguishable
  from the best; the deep-learning models should survive if the DL augmentation helps.
- **Diebold-Mariano vs the flagship** — `DM < 0` with small `p` means the flagship
  (SV-LT-RECH-SRN) significantly beats that baseline.
- Compare the **DL vs no-DL** rows within each backbone to read off the deep-learning
  effect; compare **GARCH-RECH vs SV-LT-RECH** to read off the backbone effect.

---

## 6. Notes

- SV models use Sequential Monte Carlo; runtime scales with `smcN × smcM × T`. Start
  with the defaults, then scale up `smcN` for publication-quality posteriors.
- Results are seeded (`opts.seed`, default `20260516`) and reproducible.
- The standalone files are verbatim flattenings of the tested `+package` code; running
  `runtests('tests')` once in your environment confirms the shared engine.
