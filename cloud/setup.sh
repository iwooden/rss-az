#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Lambda Labs GH200 instance setup for rss-az
#
# Usage (from a fresh instance):
#   export GH_TOKEN="ghp_..."   # GitHub personal access token (repo scope)
#   curl -sL <raw-url-to-this-script> | bash
#   # — or —
#   GH_TOKEN="ghp_..." bash cloud/setup.sh
#
# What this does:
#   1. Configures git HTTPS auth via GH_TOKEN (no SSH keys needed)
#   2. Installs system deps (Python 3.12, Ruby, build tools)
#   3. Clones the repo
#   4. Creates venv, installs Python packages + CUDA PyTorch
#   5. Builds Cython extensions
#   6. Runs the test suite
#   7. Prints a ready-to-train command
# ============================================================================

REPO_URL="https://github.com/iwooden/rss-az.git"
REPO_DIR="$HOME/rss-az"
PYTHON=python3.12

# --- Colors for status output ---
info()  { echo -e "\033[1;34m==>\033[0m \033[1m$*\033[0m"; }
ok()    { echo -e "\033[1;32m==>\033[0m \033[1m$*\033[0m"; }
fail()  { echo -e "\033[1;31m==>\033[0m \033[1m$*\033[0m"; exit 1; }

# ============================================================================
# 1. GitHub authentication
# ============================================================================
if [ -z "${GH_TOKEN:-}" ]; then
    fail "GH_TOKEN not set. Export a GitHub PAT with repo scope:\n    export GH_TOKEN=\"ghp_...\""
fi

info "Configuring git credential helper for HTTPS auth"
git config --global credential.helper store
# Write credential so git clone picks it up (no interactive prompt)
mkdir -p "$HOME/.config/git"
echo "https://oauth2:${GH_TOKEN}@github.com" > "$HOME/.git-credentials"
chmod 600 "$HOME/.git-credentials"

# ============================================================================
# 2. System dependencies
# ============================================================================
info "Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    software-properties-common \
    build-essential \
    ruby \
    > /dev/null

# Python 3.12 — Lambda instances ship 3.10/3.11; add deadsnakes if needed
if ! command -v "$PYTHON" &>/dev/null; then
    info "Adding deadsnakes PPA for Python 3.12"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
fi
sudo apt-get install -y -qq \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    > /dev/null

$PYTHON --version

# ============================================================================
# 2b. Verify /dev/shm is large enough for multi-worker training
# ============================================================================
SHM_SIZE_MB=$(df --output=size -BM /dev/shm 2>/dev/null | tail -1 | tr -d ' M')
if [ -n "$SHM_SIZE_MB" ] && [ "$SHM_SIZE_MB" -lt 1024 ]; then
    echo "WARNING: /dev/shm is only ${SHM_SIZE_MB}MB. Multi-worker training"
    echo "  uses shared memory tensors and may fail. Resize with:"
    echo "  sudo mount -o remount,size=8G /dev/shm"
fi

# ============================================================================
# 2c. Check architecture (GH200 is aarch64)
# ============================================================================
ARCH=$(uname -m)
info "Architecture: $ARCH"
if [ "$ARCH" = "aarch64" ]; then
    info "aarch64 detected — PyTorch cu124 wheels should have ARM support."
    info "If pip install torch fails, fall back to Lambda Stack's system PyTorch:"
    info "  python3.12 -m venv --system-site-packages .venv"
fi

# ============================================================================
# 3. Clone repository
# ============================================================================
if [ -d "$REPO_DIR" ]; then
    info "Repo already exists at $REPO_DIR — pulling latest"
    git -C "$REPO_DIR" pull --ff-only
else
    info "Cloning $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# ============================================================================
# 4. Python venv + dependencies
# ============================================================================
info "Creating Python 3.12 venv"
$PYTHON -m venv .venv
source .venv/bin/activate

info "Upgrading pip"
pip install --upgrade pip -q

info "Running install.sh (CUDA mode)"
./install.sh cuda

# ============================================================================
# 5. Verify build
# ============================================================================
info "Clean-building Cython extensions"
python setup.py clean
python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true

# ============================================================================
# 6. Run tests
# ============================================================================
info "Running test suite"
pytest tests/ -x -q

# ============================================================================
# 7. Verify GPU is visible
# ============================================================================
info "GPU check"
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name()}')
    cap = torch.cuda.get_device_capability()
    print(f'Compute capability: {cap[0]}.{cap[1]}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ============================================================================
# Done
# ============================================================================
ok "Setup complete! Start training with:"
echo ""
echo "  cd $REPO_DIR && source .venv/bin/activate"
echo "  python -m train --device cuda --num-workers 4 --search-batch-size 8"
echo ""
