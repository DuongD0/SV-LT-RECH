#!/usr/bin/env bash
# run_matlab.sh — launch MATLAB with workarounds for too-new Linux / glibc + WSL2.
#
# MATLAB R2026a is validated only on Ubuntu 22.04/24.04, Debian 12/13, RHEL 8/9,
# SLES/SLED 15 (glibc >= 2.28); WSL2 is officially unsupported. On a newer OS
# (e.g. Ubuntu 26.04 / glibc 2.43) the desktop segfaults at startup because
# glibc 2.35+ restartable sequences (rseq) clash with MATLAB's bundled tcmalloc
# (MathWorks bug 2632298). This wrapper applies the documented escape hatches.
#
# Usage:
#   ./run_matlab.sh                              # GUI desktop
#   ./run_matlab.sh -batch "addpath(pwd); runtests('tests')"   # headless
#   ./run_matlab.sh -nodisplay -batch "run('scripts/run_stage4_demo.m')"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer a licensed system MATLAB on PATH; fall back to the bundled tree.
if command -v matlab >/dev/null 2>&1; then
    MATLAB_BIN="$(command -v matlab)"
elif [ -x "$HERE/MATLAB_R2026a/bin/matlab" ]; then
    MATLAB_BIN="$HERE/MATLAB_R2026a/bin/matlab"
else
    echo "run_matlab.sh: no 'matlab' on PATH and no $HERE/MATLAB_R2026a/bin/matlab" >&2
    exit 1
fi

# (1) Disable glibc restartable sequences — fixes the startup segfault on glibc 2.35+.
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.pthread.rseq=0}"
# (2) WSL2 has no native GPU GL stack — force software OpenGL for GUI sessions.
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

echo "run_matlab.sh: $MATLAB_BIN  (GLIBC_TUNABLES=$GLIBC_TUNABLES)" >&2
exec "$MATLAB_BIN" "$@"
