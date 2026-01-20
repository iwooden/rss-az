# External Integrations

**Analysis Date:** 2026-01-20

## Overview

This codebase has minimal external integrations. It is a self-contained high-performance game engine for "Rolling Stock Stars" optimized for AlphaZero-style self-play training. All state management and game logic is internal.

## APIs & External Services

**Not detected** - No external API calls or third-party services integrated.

The engine operates entirely within the process:
- Game simulation runs synchronously in compiled Cython code
- State is managed as internal float32 arrays
- No network communication
- No cloud services or webhooks

## Data Storage

**Databases:**
- Not applicable - No persistent database integration

**File Storage:**
- Local filesystem only
- Game state exists only in memory during execution
- Benchmarking results printed to stdout

**Caching:**
- Not applicable - All state is in-process memory

## Authentication & Identity

**Not applicable** - Single-player/self-play training engine with no user authentication or identity management.

## Monitoring & Observability

**Error Tracking:**
- Not detected - No error reporting service

**Logs:**
- stdout/stderr only
- Benchmark output to console
- Custom clean command prints to stdout

**Debugging:**
- Cython annotation files can be generated (currently disabled in `setup.py`, `annotate=False`)
- embedsignature enabled for function signature introspection

## CI/CD & Deployment

**Hosting:**
- Local development machine or GPU-enabled training server
- No cloud deployment infrastructure detected

**CI Pipeline:**
- Not detected - No CI/CD configuration files (.github/workflows, .gitlab-ci.yml, etc.)

## Environment Configuration

**Required env vars:**
- None detected - Configuration is hard-coded or passed as command-line arguments

**Example usage:**
```bash
python setup.py benchmark --num-games=5000 --num-players=6
```

**Secrets location:**
- Not applicable - No secrets management required

## Webhooks & Callbacks

**Incoming:**
- Not applicable - No incoming webhooks

**Outgoing:**
- Not applicable - No outgoing webhooks

## GPU/Hardware Integration

**Hardware Acceleration:**
- PyTorch with ROCm 6.4.2 for AMD GPU support
- Cython compiled to native code for CPU optimization
- No auto-detection of hardware; ROCm hard-coded at dependency level

## Dependencies on External State

**Complete isolation:**
- All game state is deterministic given initial player count and random seed
- State array is self-contained and can be serialized for training
- No external state lookups or RPC calls

## Performance Characteristics

**In-process execution:**
- Game simulation runs entirely within Python/Cython process
- State passed directly to PyTorch as NumPy arrays (zero-copy)
- Benchmark measures games per minute on local hardware

**Scaling:**
- Horizontal scaling via multiple independent processes (distributed training coordination external)
- Vertical scaling limited by GPU memory and CPU cores

---

*Integration audit: 2026-01-20*
