# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance Cython game engine for "Rolling Stock Stars" board game, optimized for AlphaZero-style self-play training. Game state is stored as a single contiguous float32 array that can be passed directly to PyTorch without serialization overhead.

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
