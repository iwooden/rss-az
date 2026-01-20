# Technology Stack

**Analysis Date:** 2026-01-20

## Languages

**Primary:**
- Python 3.12.3 - Game engine core, setup and scripting
- Cython 3.2.4 - High-performance compiled extensions for game logic

**Secondary:**
- C (generated from Cython) - Low-level performance-critical code paths

## Runtime

**Environment:**
- CPython 3.12.3 running on Linux (WSL2)
- Virtual environment located at `.venv/`

**Package Manager:**
- pip 25.3
- Lockfile: Not detected (pip freeze would be needed)

## Frameworks

**Core:**
- Cython 3.2.4 - Compiles Python to C for performance-critical game simulation
- NumPy 2.4.0 - Numerical arrays for game state representation
- PyTorch 2.6.0+rocm6.4.2 - Neural network training (AlphaZero-style self-play)
- PyTorch extensions:
  - torchaudio 2.6.0+rocm6.4.2
  - torchvision 0.21.0+rocm6.4.2
  - pytorch-triton-rocm 3.2.0+rocm6.4.2 - GPU compilation for NVIDIA/ROCm backends

**Testing:**
- pytest 9.0.2 - Test runner (referenced in CLAUDE.md)

**Build/Dev:**
- setuptools 80.9.0 - Python package building
- Pillow 12.1.0 - Image processing (optional dependency)

## Key Dependencies

**Critical:**
- numpy 2.4.0 - Game state stored as contiguous float32 arrays, direct PyTorch compatibility
- Cython 3.2.4 - Compiles game logic to native code for high-performance self-play training
- torch 2.6.0+rocm6.4.2 - AlphaZero neural network training on GPU (ROCm backend for AMD cards)

**Infrastructure:**
- setuptools 80.9.0 - Extension module compilation and package distribution
- networkx 3.6.1 - Graph analysis utilities
- sympy 1.3.0 - Symbolic mathematics

## Compiler Configuration

**Cython Optimization Directives:**
Located in `setup.py`:
```
language_level: '3'
boundscheck: False           # Disable array bounds checking
wraparound: False            # Disable negative array indexing
cdivision: True              # Use C-level division (faster)
initializedcheck: False      # Skip uninitialized variable checks
nonecheck: False             # Skip None type checks
overflowcheck: False         # Skip integer overflow checks
embedsignature: True         # Include function signatures in docstrings
```

**Extension Module Discovery:**
- Searches `helpers/`, `phases/`, and `core/` directories for `.pyx` files
- Also includes root-level `.pyx` files
- Includes NumPy headers via `np.get_include()`
- Defines `NPY_NO_DEPRECATED_API` macro for NumPy 1.7+ compatibility

## Configuration

**Environment:**
- Python version hardcoded via shebang and venv config to 3.12.3
- No `.env` files detected
- ROCm 6.4.2 enabled for GPU acceleration

**Build:**
- `setup.py` - Main build configuration
- `.pyx` files - Cython source files (game logic)
- `.pxd` files - Cython header files (type definitions and cimports)
- `.c` files - Generated from Cython (not committed, auto-generated)
- `.so` files - Compiled extension modules (not committed, auto-generated)

## Build Commands

```bash
# Build Cython extensions (required before running any Python code)
python setup.py build_ext --inplace

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_invest.py -v

# Clean build artifacts (.c, .so, .html, build/, *.egg-info)
python setup.py clean

# Benchmark: measure games per minute (requires build first)
python setup.py benchmark                          # 1000 games, 3 players
python setup.py benchmark --num-games=5000 --num-players=6
```

## Platform Requirements

**Development:**
- CPython 3.12.3 with venv support
- C compiler (for Cython-generated C code)
- NumPy development headers
- Linux/WSL2 environment
- GPU support (AMD/ROCm 6.4.2)

**Production:**
- Same as development (JIT compilation occurs at build time)
- Target: AlphaZero-style self-play training on GPU

## Module Organization

**Compiled Packages:**
- `core/` - Game state and data (state.pyx, data.pyx)
- `helpers/` - Utility functions
- `entities/` - Game entities (player, corp, company, market, etc.)
- `phases/` - Game phase logic
- `actions.pyx` - Action space and decoding at root level

**Entry Points:**
- `setup.py` - Package build and custom commands
- Root-level `.pyx` files compiled as top-level modules

---

*Stack analysis: 2026-01-20*
