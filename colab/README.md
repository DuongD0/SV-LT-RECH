# Colab Stage 1 study — operator instructions

This folder is a **self-contained Colab port** of the Phase 3 Stage 1
simulation study for SV-LT-RECH. It lives separately from the local
MATLAB pipeline so the two can evolve independently.

The point: get a publication-quality Stage 1 parameter-recovery study
running on a Colab VM, with all the outputs needed to feed Phase 4
(LSTM/GRU cells, RealRECH, GARCH baselines) without re-running anything.

## What's in here

```
colab/
├── README.md                              ← you are here
├── PHASE_4_HANDOFF.md                     ← spec of what comes back
├── install_matlab.sh                      ← MATLAB install on Colab via mpm
├── stage1_colab.ipynb                     ← Colab notebook (open this)
└── matlab_src/                            ← Stage-1-only MATLAB code mirror
    ├── +cells/srn.m
    ├── +diagnostics/{posteriorTrace,rHat,mcseLogML,relabelSignSwitch}.m
    ├── +experiments/runStage1Colab.m      ← resumable Colab runner
    ├── +inference/+mcmc/rwMetropolis.m
    ├── +inference/+pf/{bootstrap,resampleSystematic,resampleStratified}.m
    ├── +inference/+smc/{likelihoodAnneal,adaptiveTemperature,ess}.m
    ├── +models/{Model,SV,SVt,SVLT,SVLTRECH}.m
    ├── +priors/logPrior{SV,SVt,SVLT,SVLTRECH}.m
    ├── +utils/{logsumexp,log1pexp,reproducibility,leverageCholesky,leverageOcsn,loadConfigColab}.m
    ├── config/
    │   ├── priors.yaml
    │   ├── smc_defaults.yaml
    │   └── experiments/
    │       ├── stage1_colab.json          ← FULL study
    │       └── stage1_colab_quick.json    ← sanity variant (~30 min)
    └── scripts/
        ├── run_stage1_colab.m             ← entry script
        └── package_results.m              ← bundles outputs for download
```

The `matlab_src/` mirror **does not** include the Phase 2 data fetchers
(`+data/`), the evaluation suite (`+eval/`), or the local
PyYAML-dependent `+utils/loadConfig.m`. Those aren't needed for the
simulation study and reduce the bundle size + license surface.

Configs are shipped as **JSON** (not YAML) because `readstruct` in
the Colab MATLAB release doesn't yet support YAML and we want zero
runtime dependencies. The schema is identical to the local YAML —
edit `stage1_colab.json` directly if you want to tweak.

## One-time setup

You need a MathWorks account with online licensing entitlements (the
free Community license works for non-commercial / academic use).

## How to use this

### Step 1. Bundle this folder

From the project root on your local machine:

```bash
cd /path/to/finance_eng
tar -czvf finance_eng_colab.tar.gz -C colab .
```

That produces a `finance_eng_colab.tar.gz` you'll upload to Colab.

### Step 2. Open the notebook in Colab

1. Go to https://colab.research.google.com/.
2. **File → Upload notebook** → pick `colab/stage1_colab.ipynb`.
3. **Runtime → Change runtime type** → CPU.
4. In the Files sidebar, **upload** `finance_eng_colab.tar.gz` to `/content/`.
5. Run the cells top-to-bottom.

### Step 3. Activate MATLAB

License activation is **interactive**, so it has to run in a Colab
terminal (Tools → Terminal), not a notebook `!` cell. In that terminal:

```bash
/usr/local/MATLAB/R2026a/bin/matlab -nodesktop -licmode onlinelicensing
```

1. Enter your MathWorks account email when prompted.
2. Get a one-time password from <https://www.mathworks.com/mwa/otp>.
3. Paste the OTP into the MATLAB prompt.
4. Type `exit` to close that MATLAB session.

License is cached for the 12-hour Colab VM lifetime. The notebook's
cell 6 (verification probe) confirms it's live before you launch the
study. Same flow as the MathWorks blog post; only the release name
changes (R2025a → R2026a).

### Step 4. Sanity-check with the QUICK variant

Run cell 4 (~30 min). At the end, cell 5 prints `gate_summary.json`.
Expect the gates to FAIL — the quick variant is intentionally
undersized. What you're checking: it ran to completion, produced
sensible numbers, no MATLAB errors in the log.

### Step 5. Launch the FULL study

Run cell 6. This is the multi-hour run. The runner is **resumable**:

- After each replicate, partial CSVs + posterior `.mat` files are
  written to `/content/finance_eng_colab/results/stage1_colab/`.
- If the Colab VM expires (12-hour limit), re-attach a new VM, re-run
  the install + activation cells, and re-run cell 6. It picks up from
  the last completed replicate (seed-deterministic, so results are
  bit-identical to a single-session run).

You can `tail -f` the run log (cell 7) at any time to see progress.

### Step 6. Package + download

Run the two final cells. They tar everything up and trigger a browser
download of `phase4_handoff_<runId>_<timestamp>.tar.gz`.

### Step 7. Ship back to me

DM / upload the archive back to the conversation. I'll unpack it,
verify each §4.5 acceptance gate, and use the per-replicate posterior
particles to warm-start the Phase 4 model variants.

See `PHASE_4_HANDOFF.md` for the exact archive contents I expect.

## Troubleshooting

### "matlab: command not found" after install

Re-run the install cell — the `ln -sf` step at the end is what makes
`matlab` visible on PATH. If it still fails, run `which mpm` to confirm
the install ran in `/content/finance_eng_colab/`.

### License activation hangs

Open a Colab terminal (Tools → Terminal) and run

```bash
/usr/local/MATLAB/R2026a/bin/matlab -nodesktop -licmode onlinelicensing
```

That triggers the OTP prompt interactively. Once you finish there, the
cached license works for all subsequent `matlab -batch` calls — **as
long as you pass `-licmode onlinelicensing` on every launch**.

### License Error -1.2 "Unable to find a license for MATLAB"

You launched MATLAB without `-licmode onlinelicensing`, so it tried
file-based licensing and found nothing. Every notebook cell in this
runbook already passes that flag; if you're running an ad-hoc command,
mirror the pattern:

```bash
matlab -licmode onlinelicensing -batch "your_code"
```

If you DID pass the flag and still see -1.2, you haven't completed the
OTP sign-in yet — go back to the Step 3 terminal and finish it.

### "yaml" not supported by readstruct

You shouldn't see this — the Colab path uses JSON configs only. If
you see this error, you're calling the LOCAL pipeline's
`utils.loadConfig` (which requires PyYAML) instead of
`utils.loadConfigColab`. The Colab runner always uses the latter.

### Resume reads stale rows

If the CSV got corrupted mid-write (rare, but possible on Colab VM
preemption), delete `results/stage1_colab/recovery.csv` to force a
clean restart. The seed-deterministic DGP means the data simulated for
each replicate is identical to before.

### Colab session expires too quickly

Use Colab Pro for 24h sessions if available; otherwise plan for 2-3
sessions for the full study (the runner resumes cleanly).
