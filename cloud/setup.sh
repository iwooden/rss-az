#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Cloud GPU instance setup for rss-az (the RTX 5090; first written for a GH200)
#
# The 1889 run's steps after this are in 1889/model-design.md, Section 11.
#
# Usage (from a fresh instance):
#   export GH_TOKEN="ghp_..."   # GitHub personal access token (repo scope)
#   curl -sL <raw-url-to-this-script> | bash
#   # — or —
#   GH_TOKEN="ghp_..." bash cloud/setup.sh
#   # — or, on a pod whose disk that lasts is a volume —
#   GH_TOKEN="ghp_..." REPO_DIR=/workspace/rss-az bash cloud/setup.sh
#
# What this does:
#   1. Configures git HTTPS auth via GH_TOKEN (no SSH keys needed)
#   2. Installs system deps (Python 3.12, Ruby, build tools)
#   3. Clones the repo, on $REPO_BRANCH (default az-1889), to $REPO_DIR
#   4. Creates venv, installs Python packages + CUDA PyTorch
#   5. Builds Cython extensions, with their assertions: a --release build strips
#      them, and the tests that feed the engine invalid input then crash
#   6. Runs the test suite
#   7. Prints a ready-to-train command
# ============================================================================

REPO_URL="https://github.com/iwooden/rss-az.git"
REPO_DIR="${REPO_DIR:-$HOME/rss-az}"
REPO_BRANCH="${REPO_BRANCH:-az-1889}"
PYTHON=python3.12
# A container's root has no sudo, and needs none.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO=sudo; fi

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
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    software-properties-common \
    build-essential \
    ruby \
    > /dev/null

# Python 3.12 — Lambda instances ship 3.10/3.11; add deadsnakes if needed
if ! command -v "$PYTHON" &>/dev/null; then
    info "Adding deadsnakes PPA for Python 3.12"
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    $SUDO apt-get update -qq
fi
$SUDO apt-get install -y -qq \
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
    info "aarch64 detected — check that install.sh's CUDA wheels have ARM support."
    info "If pip install torch fails, fall back to Lambda Stack's system PyTorch:"
    info "  python3.12 -m venv --system-site-packages .venv"
fi

# ============================================================================
# 3. Clone repository
# ============================================================================
# The submodule is left out: only tests and 18xx.games tooling read it, and they
# skip without it. g1889's tables are committed; Ruby only regenerates them.
if [ -d "$REPO_DIR" ]; then
    info "Repo already exists at $REPO_DIR — pulling $REPO_BRANCH"
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH"
else
    info "Cloning $REPO_URL ($REPO_BRANCH)"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
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
# Tests that need the submodule skip. A failure stops nothing here, so the GPU check
# below still runs: read any failures before launching.
info "Running test suite"
pytest tests/ -q || info "Tests failed: read the list above before launching."

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
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ============================================================================
# Done
# ============================================================================
ok "Setup complete! Next, per 1889/model-design.md Section 11:"
echo ""
echo "  cd $REPO_DIR && source .venv/bin/activate"
echo "  python cloud/profile_1889.py"
echo "  TORCH_LOGS=recompiles python -m train --config train_configs/1889-v1.json"
echo ""
