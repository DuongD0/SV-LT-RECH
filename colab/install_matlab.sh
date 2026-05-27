#!/usr/bin/env bash
# install_matlab.sh -- one-shot MATLAB R2026a install on Google Colab via mpm.
#
# Mirrors the MathWorks blog post (R2025a -> R2026a substituted):
#   https://blogs.mathworks.com/matlab/2025/06/27/using-matlab-on-google-colab/
#
# Default mpm destination is /usr/local/MATLAB/$RELEASE -- we keep that so the
# launch command matches the blog verbatim:
#   /usr/local/MATLAB/R2026a/bin/matlab -nodesktop -licmode onlinelicensing
#
# Products: MATLAB + Statistics_and_Machine_Learning_Toolbox. Stage 1 uses
# betarnd / trnd / exprnd / quantile from Stats; everything else is base MATLAB.

set -euo pipefail

RELEASE="${MATLAB_RELEASE:-R2026a}"
PRODUCTS="${MATLAB_PRODUCTS:-MATLAB Statistics_and_Machine_Learning_Toolbox}"

echo "[install_matlab] release:  $RELEASE"
echo "[install_matlab] products: $PRODUCTS"

if [ ! -x ./mpm ]; then
    echo "[install_matlab] downloading mpm..."
    wget -qO mpm https://www.mathworks.com/mpm/glnxa64/mpm
    chmod +x mpm
fi

./mpm install --release="$RELEASE" --products $PRODUCTS

ln -sf "/usr/local/MATLAB/$RELEASE/bin/matlab" /usr/local/bin/matlab

echo
echo "[install_matlab] installed at /usr/local/MATLAB/$RELEASE"
echo "[install_matlab] symlink:    /usr/local/bin/matlab"
echo
echo "Next -- activate the license in a Colab TERMINAL (Tools -> Terminal):"
echo "  /usr/local/MATLAB/$RELEASE/bin/matlab -nodesktop -licmode onlinelicensing"
echo "Follow the OTP prompt at https://www.mathworks.com/mwa/otp"
echo
echo "IMPORTANT: every subsequent 'matlab -batch ...' call must also pass"
echo "  -licmode onlinelicensing"
echo "or it will fail with 'License Error -1.2: Unable to find a license'."
