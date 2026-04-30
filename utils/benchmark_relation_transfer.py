#!/usr/bin/env python3
"""Benchmark dense relation-plane transfer vs sparse coordinate materialization.

This is intended for cloud GPU runs that mirror the eval-server input path:

  dense:  copy (B, R, N, N) uint8 relation planes from pinned CPU to GPU
  sparse: copy (B, MAX_RELATIONS, 3) uint8 coordinates, then materialize the
          dense GPU relation tensor with zero_ + flattened index_fill_

Examples:

  PYTHONPATH=. .venv/bin/python utils/benchmark_relation_transfer.py
  PYTHONPATH=. .venv/bin/python utils/benchmark_relation_transfer.py --json
  PYTHONPATH=. .venv/bin/python utils/benchmark_relation_transfer.py \
      --iters 2000 --warmup 200 --max-relations 256
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class ResolvedConstant:
    value: int
    source: str


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    cuda_ms_per_iter: float
    wall_ms_per_iter: float
    h2d_bytes_per_iter: int
    effective_h2d_gib_s_cuda: float | None
    notes: str


def _resolve_num_tokens(num_players: int, override: int | None) -> ResolvedConstant:
    if override is not None:
        return ResolvedConstant(override, "--num-tokens")
    try:
        from core.token_data import get_num_tokens

        return ResolvedConstant(int(get_num_tokens(num_players)), "core.token_data")
    except Exception:
        return ResolvedConstant(num_players + 54, "fallback:num_players+54")


def _resolve_num_relations(override: int | None) -> ResolvedConstant:
    if override is not None:
        return ResolvedConstant(override, "--num-relations")
    try:
        from core.attention_relations import NUM_ATTENTION_RELATIONS

        return ResolvedConstant(
            int(NUM_ATTENTION_RELATIONS),
            "core.attention_relations",
        )
    except Exception:
        return ResolvedConstant(10, "fallback:10")


def _format_bytes(n: int) -> str:
    mib = n / (1024 * 1024)
    return f"{n:,} bytes ({mib:.3f} MiB)"


def _effective_gib_s(num_bytes: int, ms: float) -> float | None:
    if num_bytes <= 0 or ms <= 0.0:
        return None
    return (num_bytes / (1024 ** 3)) / (ms / 1000.0)


def _time_cuda_op(
    *,
    torch_mod,
    fn: Callable[[], None],
    iters: int,
    warmup: int,
    h2d_bytes_per_iter: int,
    name: str,
    notes: str,
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    torch_mod.cuda.synchronize()

    start_event = torch_mod.cuda.Event(enable_timing=True)
    end_event = torch_mod.cuda.Event(enable_timing=True)

    wall_start = perf_counter()
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    end_event.synchronize()
    wall_end = perf_counter()

    cuda_ms = float(start_event.elapsed_time(end_event)) / iters
    wall_ms = (wall_end - wall_start) * 1000.0 / iters
    return BenchmarkResult(
        name=name,
        cuda_ms_per_iter=cuda_ms,
        wall_ms_per_iter=wall_ms,
        h2d_bytes_per_iter=h2d_bytes_per_iter,
        effective_h2d_gib_s_cuda=_effective_gib_s(h2d_bytes_per_iter, cuda_ms),
        notes=notes,
    )


def _make_sparse_coords(
    *,
    torch_mod,
    batch_size: int,
    max_relations: int,
    active_relations: int,
    num_relations: int,
    num_tokens: int,
    pin_memory: bool,
    seed: int,
):
    generator = torch_mod.Generator(device="cpu")
    generator.manual_seed(seed)

    coords = torch_mod.zeros(
        batch_size,
        max_relations,
        3,
        dtype=torch_mod.uint8,
        pin_memory=pin_memory,
    )
    if active_relations == 0:
        return coords

    # Real relation edges never use the padded sentinel (0, 0, 0). Generate
    # token coordinates from [1, N) to keep the sentinel clear in the benchmark.
    rel = torch_mod.randint(
        0,
        num_relations,
        (batch_size, active_relations),
        generator=generator,
        dtype=torch_mod.int64,
    ).to(torch_mod.uint8)
    row = torch_mod.randint(
        1,
        num_tokens,
        (batch_size, active_relations),
        generator=generator,
        dtype=torch_mod.int64,
    ).to(torch_mod.uint8)
    col = torch_mod.randint(
        1,
        num_tokens,
        (batch_size, active_relations),
        generator=generator,
        dtype=torch_mod.int64,
    ).to(torch_mod.uint8)

    coords[:, :active_relations, 0].copy_(rel)
    coords[:, :active_relations, 1].copy_(row)
    coords[:, :active_relations, 2].copy_(col)
    return coords


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark needs a GPU.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit(f"expected a CUDA device, got {device!s}")

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_relations <= 0:
        raise SystemExit("--max-relations must be positive")
    active_relations = (
        args.max_relations
        if args.active_relations is None
        else args.active_relations
    )
    if active_relations < 0 or active_relations > args.max_relations:
        raise SystemExit("--active-relations must be between 0 and --max-relations")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    token_const = _resolve_num_tokens(args.num_players, args.num_tokens)
    relation_const = _resolve_num_relations(args.num_relations)
    num_tokens = token_const.value
    num_relations = relation_const.value

    if num_tokens >= 255:
        raise SystemExit(
            f"num_tokens={num_tokens} does not fit the proposed uint8 encoding"
        )
    if num_relations >= 255:
        raise SystemExit(
            f"num_relations={num_relations} does not fit the proposed uint8 encoding"
        )

    pin_memory = not args.no_pin
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    batch_size = args.batch_size
    max_relations = args.max_relations

    dense_h2d_bytes = batch_size * num_relations * num_tokens * num_tokens
    sparse_h2d_bytes = batch_size * max_relations * 3

    dense_cpu = torch.empty(
        batch_size,
        num_relations,
        num_tokens,
        num_tokens,
        dtype=torch.uint8,
        pin_memory=pin_memory,
    )
    dense_cpu.random_(0, 2)
    dense_gpu = torch.empty_like(dense_cpu, device=device)

    sparse_cpu = _make_sparse_coords(
        torch_mod=torch,
        batch_size=batch_size,
        max_relations=max_relations,
        active_relations=active_relations,
        num_relations=num_relations,
        num_tokens=num_tokens,
        pin_memory=pin_memory,
        seed=args.seed or 0,
    )
    sparse_gpu = torch.empty_like(sparse_cpu, device=device)

    materialized_gpu = torch.empty(
        batch_size,
        num_relations,
        num_tokens,
        num_tokens,
        dtype=torch.uint8,
        device=device,
    )
    materialized_flat = materialized_gpu.reshape(-1)

    flat_idx = torch.empty(
        batch_size,
        max_relations,
        dtype=torch.long,
        device=device,
    )
    flat_idx_flat = flat_idx.reshape(-1)
    tmp_idx = torch.empty_like(flat_idx)
    batch_offsets = (
        torch.arange(batch_size, dtype=torch.long, device=device)
        * (num_relations * num_tokens * num_tokens)
    ).reshape(batch_size, 1)
    sentinel_flat = batch_offsets.reshape(-1).contiguous()
    relation_stride = num_tokens * num_tokens

    def dense_copy() -> None:
        dense_gpu.copy_(dense_cpu, non_blocking=pin_memory)

    def sparse_copy() -> None:
        sparse_gpu.copy_(sparse_cpu, non_blocking=pin_memory)

    def materialize_sparse_gpu_only() -> None:
        # Build flattened dense indices:
        #   flat = (((batch * R + relation) * N + row) * N + col)
        # Preallocated long buffers avoid per-batch index tensor allocations.
        flat_idx.copy_(sparse_gpu[..., 0])
        flat_idx.mul_(relation_stride)
        tmp_idx.copy_(sparse_gpu[..., 1])
        flat_idx.add_(tmp_idx, alpha=num_tokens)
        tmp_idx.copy_(sparse_gpu[..., 2])
        flat_idx.add_(tmp_idx)
        flat_idx.add_(batch_offsets)

        materialized_flat.zero_()
        materialized_flat.index_fill_(0, flat_idx_flat, 1)
        materialized_flat.index_fill_(0, sentinel_flat, 0)

    def sparse_copy_and_materialize() -> None:
        sparse_gpu.copy_(sparse_cpu, non_blocking=pin_memory)
        materialize_sparse_gpu_only()

    # Put coordinates on device before timing the GPU-only materialization path.
    sparse_gpu.copy_(sparse_cpu, non_blocking=pin_memory)
    torch.cuda.synchronize()

    results = [
        _time_cuda_op(
            torch_mod=torch,
            fn=dense_copy,
            iters=args.iters,
            warmup=args.warmup,
            h2d_bytes_per_iter=dense_h2d_bytes,
            name="dense_h2d_copy",
            notes="current uint8 (B,R,N,N) relation transfer",
        ),
        _time_cuda_op(
            torch_mod=torch,
            fn=sparse_copy,
            iters=args.iters,
            warmup=args.warmup,
            h2d_bytes_per_iter=sparse_h2d_bytes,
            name="sparse_h2d_copy",
            notes="uint8 (B,MAX_RELATIONS,3) coordinate transfer only",
        ),
        _time_cuda_op(
            torch_mod=torch,
            fn=materialize_sparse_gpu_only,
            iters=args.iters,
            warmup=args.warmup,
            h2d_bytes_per_iter=0,
            name="sparse_materialize_gpu_only",
            notes="zero dense tensor + build flat long indices + index_fill_",
        ),
        _time_cuda_op(
            torch_mod=torch,
            fn=sparse_copy_and_materialize,
            iters=args.iters,
            warmup=args.warmup,
            h2d_bytes_per_iter=sparse_h2d_bytes,
            name="sparse_h2d_plus_materialize",
            notes="full proposed replacement for dense relation copy",
        ),
    ]

    torch.cuda.synchronize()

    config = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size": batch_size,
        "num_players": args.num_players,
        "num_tokens": num_tokens,
        "num_tokens_source": token_const.source,
        "num_relations": num_relations,
        "num_relations_source": relation_const.source,
        "max_relations": max_relations,
        "active_relations": active_relations,
        "pin_memory": pin_memory,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    by_name = {result.name: result for result in results}
    dense_result = by_name["dense_h2d_copy"]
    sparse_full_result = by_name["sparse_h2d_plus_materialize"]
    cuda_delta_ms = (
        sparse_full_result.cuda_ms_per_iter
        - dense_result.cuda_ms_per_iter
    )
    wall_delta_ms = (
        sparse_full_result.wall_ms_per_iter
        - dense_result.wall_ms_per_iter
    )

    payload: dict[str, object] = {
        "config": config,
        "bytes": {
            "dense_relation_h2d_per_batch": dense_h2d_bytes,
            "sparse_relation_h2d_per_batch": sparse_h2d_bytes,
            "saved_relation_h2d_per_batch": dense_h2d_bytes - sparse_h2d_bytes,
            "relation_h2d_reduction": (
                1.0 - (sparse_h2d_bytes / dense_h2d_bytes)
                if dense_h2d_bytes
                else 0.0
            ),
        },
        "comparison": {
            "sparse_full_minus_dense_cuda_ms": cuda_delta_ms,
            "sparse_full_minus_dense_wall_ms": wall_delta_ms,
            "sparse_full_vs_dense_cuda_speedup": (
                dense_result.cuda_ms_per_iter / sparse_full_result.cuda_ms_per_iter
                if sparse_full_result.cuda_ms_per_iter > 0.0
                else None
            ),
            "sparse_full_vs_dense_wall_speedup": (
                dense_result.wall_ms_per_iter / sparse_full_result.wall_ms_per_iter
                if sparse_full_result.wall_ms_per_iter > 0.0
                else None
            ),
        },
        "results": [asdict(result) for result in results],
    }
    return payload


def print_human(payload: dict[str, object]) -> None:
    config = payload["config"]
    bytes_info = payload["bytes"]
    comparison = payload["comparison"]
    results = payload["results"]

    print("Configuration")
    print(f"  device             : {config['device']} ({config['device_name']})")
    print(f"  torch/cuda         : {config['torch_version']} / {config['cuda_version']}")
    print(f"  batch_size         : {config['batch_size']}")
    print(
        "  num_tokens         : "
        f"{config['num_tokens']} ({config['num_tokens_source']})"
    )
    print(
        "  num_relations      : "
        f"{config['num_relations']} ({config['num_relations_source']})"
    )
    print(f"  max_relations      : {config['max_relations']}")
    print(f"  active_relations   : {config['active_relations']}")
    print(f"  pin_memory         : {config['pin_memory']}")
    print(f"  warmup / iters     : {config['warmup']} / {config['iters']}")
    print()

    dense_bytes = int(bytes_info["dense_relation_h2d_per_batch"])
    sparse_bytes = int(bytes_info["sparse_relation_h2d_per_batch"])
    saved_bytes = int(bytes_info["saved_relation_h2d_per_batch"])
    reduction = float(bytes_info["relation_h2d_reduction"])

    print("Relation H2D Payload")
    print(f"  dense per batch    : {_format_bytes(dense_bytes)}")
    print(f"  sparse per batch   : {_format_bytes(sparse_bytes)}")
    print(f"  saved per batch    : {_format_bytes(saved_bytes)}")
    print(f"  reduction          : {reduction * 100.0:.2f}%")
    print()

    headers = (
        "benchmark",
        "cuda ms",
        "wall ms",
        "H2D MiB",
        "H2D GiB/s",
        "notes",
    )
    rows = []
    for result in results:
        gib_s = result["effective_h2d_gib_s_cuda"]
        rows.append((
            result["name"],
            f"{result['cuda_ms_per_iter']:.4f}",
            f"{result['wall_ms_per_iter']:.4f}",
            f"{result['h2d_bytes_per_iter'] / (1024 * 1024):.3f}",
            "" if gib_s is None else f"{gib_s:.2f}",
            result["notes"],
        ))

    widths = [
        max(len(str(row[i])) for row in (headers, *rows))
        for i in range(len(headers))
    ]
    print("Results")
    print("  " + "  ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))

    print()
    print("Comparison")
    print(
        "  sparse_full - dense cuda : "
        f"{comparison['sparse_full_minus_dense_cuda_ms']:+.4f} ms/batch"
    )
    print(
        "  sparse_full - dense wall : "
        f"{comparison['sparse_full_minus_dense_wall_ms']:+.4f} ms/batch"
    )
    print(
        "  cuda speedup             : "
        f"{comparison['sparse_full_vs_dense_cuda_speedup']:.3f}x"
    )
    print(
        "  wall speedup             : "
        f"{comparison['sparse_full_vs_dense_wall_speedup']:.3f}x"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="override token count; default resolves from core.token_data",
    )
    parser.add_argument(
        "--num-relations",
        type=int,
        default=None,
        help="override relation count; default resolves from core.attention_relations",
    )
    parser.add_argument("--max-relations", type=int, default=256)
    parser.add_argument(
        "--active-relations",
        type=int,
        default=None,
        help=(
            "non-sentinel coordinates per state; default equals --max-relations. "
            "Padded rows remain (0,0,0) and are still processed."
        ),
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help="use pageable CPU tensors instead of pinned CPU tensors",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a human-readable table",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_benchmark(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
