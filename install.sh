#!/usr/bin/env bash
set -euo pipefail

# Install all Python dependencies for rss-az-cython2.
#
# Usage:
#   ./install.sh              # defaults to CPU-only PyTorch
#   ./install.sh cuda         # torch 2.14 on CUDA 13.2, what the cloud 5090 runs
#   ./install.sh rocm         # ROCm 6.2
#
# Assumes you've already created and activated a venv, or are running
# inside the project venv (e.g. .venv/bin/python install.sh won't work —
# use: source .venv/bin/activate && ./install.sh)

PLATFORM="${1:-cpu}"
TORCH_SPEC="torch"

case "$PLATFORM" in
    cpu)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        ;;
    cuda)
        TORCH_INDEX="https://download.pytorch.org/whl/cu132"
        # Blackwell needs a recent CUDA, and optim.Muon a recent torch. Pinned, as
        # the index's newest is not what the run's compile path was profiled on.
        TORCH_SPEC="torch==2.14.*"
        ;;
    rocm)
        TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"
        ;;
    *)
        echo "Unknown platform: $PLATFORM"
        echo "Usage: $0 [cpu|cuda|rocm]"
        exit 1
        ;;
esac

echo "==> Installing PyTorch for $PLATFORM"
pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"

echo "==> Installing main dependencies"
pip install \
    setuptools \
    'numpy>=2.0' \
    'Cython>=3.0' \
    'tensorboard>=2.0' \
    'rich>=13.0' \
    'matplotlib>=3.0' \
    'tqdm>=4.0' \
    'scikit-learn>=1.0' \
    'pytest>=8.0'

# captum declares numpy<2.0 but works fine with 2.x.
# --no-deps prevents it from downgrading numpy.
echo "==> Installing captum (--no-deps to preserve numpy 2.x)"
pip install --no-deps 'captum>=0.7'

echo "==> Building Cython extensions"
python setup.py build_ext --inplace

echo "==> Done"
