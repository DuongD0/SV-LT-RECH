#!/usr/bin/env bash
# setup.sh — activate every interpreter this project depends on.
#
# Usage:
#   source setup.sh
#
# After sourcing, the shell sees:
#   - the Python venv at .venv/  (numpy, pandas, yfinance, vnstock, ...)
#   - the conda env sv-lt-rech-r (R 4.5.3 + stochvol 3.2.9)
#   - SV_LT_RECH_ROOT pointing at the repo root
#   - RSCRIPT_BIN set to the conda-managed Rscript
#
# MATLAB is NOT activated here; see docs/reproducibility_checklist.md.

_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
export SV_LT_RECH_ROOT="${_SCRIPT_DIR}"

# Python venv
if [[ -f "${SV_LT_RECH_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SV_LT_RECH_ROOT}/.venv/bin/activate"
    echo "[setup] Python venv  : $(python -V 2>&1)"
else
    echo "[setup] WARNING: .venv missing. Run 'python3 -m venv .venv && pip install -r requirements.txt'."
fi

# Conda env for R
if command -v conda >/dev/null 2>&1; then
    _CONDA_BASE="$(conda info --base 2>/dev/null)"
    if [[ -n "${_CONDA_BASE}" && -f "${_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${_CONDA_BASE}/etc/profile.d/conda.sh"
        if conda env list 2>/dev/null | grep -q '^sv-lt-rech-r '; then
            conda activate sv-lt-rech-r
            echo "[setup] R (conda)    : $(R --version | head -1)"
            export RSCRIPT_BIN="$(command -v Rscript)"
        else
            echo "[setup] WARNING: conda env sv-lt-rech-r not found. Run 'conda create -n sv-lt-rech-r -c conda-forge r-base r-stochvol -y'."
        fi
    fi
fi

# MATLAB
_MATLAB_ROOT="${HOME}/MATLAB/R2026a"
if [[ -x "${_MATLAB_ROOT}/bin/matlab" ]]; then
    export PATH="${_MATLAB_ROOT}/bin:${PATH}"
    export MATLAB_ROOT="${_MATLAB_ROOT}"
    echo "[setup] MATLAB       : R2026a at ${_MATLAB_ROOT}"
elif command -v matlab >/dev/null 2>&1; then
    echo "[setup] MATLAB       : $(matlab -batch 'disp(version)' 2>/dev/null | tail -1) (from PATH)"
else
    echo "[setup] NOTE: matlab not on PATH. See docs/reproducibility_checklist.md."
fi

echo "[setup] repo root    : ${SV_LT_RECH_ROOT}"
