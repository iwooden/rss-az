"""Centralized NN evaluation server for multi-process self-play.

Each EvaluationServer runs as a separate process with its own Python GIL,
eliminating GIL contention that limited throughput when servers were threads.
The model's GPU parameter tensors are shared zero-copy via CUDA IPC
(torch.multiprocessing), so optimizer.step() in the main process is visible
to eval servers automatically.

Communication uses shared memory (torch tensors with share_memory_()):

  Worker → Server (inputs)
      states     : float32 (W, B, num_tokens, token_dim)
      phase_ids  : int8    (W, B)
      action_ids : int16   (W, B, K_MAX)   — bit-reinterpretable as uint16
      n_legals   : int16   (W, B)

  Server → Worker (outputs — sparse, already softmaxed on GPU)
      priors     : float32 (W, B, K_MAX)   — over legal actions only
      values     : float32 (W, B, num_players)   — canonical order (no rotation)

Dense per-phase logits (width ``MAX_ACTION_SIZE``) never cross the IPC
boundary: the server gathers dense logits at per-leaf ``action_ids[:n_legal]``
and softmaxes on the GPU inside the autocast region before copy-back. This
kills the old 14977-wide shared-mem logits buffer (~46 MB at 96 workers)
and the worker-side masked-softmax step outright.

**Signaling protocol:**

Request submission is lockfree via per-server uint64 bitmaps (Cython
atomics in mcts_core.pyx). Each worker atomically sets its bit in the
server's bitmap via fetch-or (release); the server atomically exchanges
the bitmap to zero (acquire) to claim all pending work in O(1). A
per-server mp.Event doorbell wakes idle servers; per-worker mp.Events
signal completion to workers.

**Multi-server concurrency:**

Each EvaluationServer owns a static partition of workers [worker_start,
worker_end) and only scans its partition's bitmap. Each server process
has its own GIL, CUDA context, and default stream, so multiple servers
truly overlap. Gather/scatter between per-worker slots and contiguous
inference buffers uses Cython nogil memcpy (no per-worker Python loop).

Workers do zero torch operations on the hot path — pure numpy writes
into the shared input buffers, a single bitmap publish + Event wait,
then numpy reads of the already-softmaxed sparse priors / canonical
values. No rotation, no masking, no softmax on the worker side.

RemoteEvaluator is the worker-side proxy that implements the same
``evaluate`` / ``evaluate_leaves`` / ``evaluate_terminal`` interface as
NNEvaluator, writing to shared memory instead of serializing over pipes.
"""

from __future__ import annotations

import copy
import queue as _queue
from time import perf_counter
from typing import Any

import numpy as np
import torch

from core.actions import (
    MAX_LEGAL_ACTIONS_PY,
    enumerate_legal_actions_py,
    get_decision_phase_py,
)
from core.token_data import TokenDataSize, get_num_tokens, get_token_data
from mcts.evaluator import BaseEvaluator
from nn.transformer import NUM_PHASES
from train.profile_stats import EvalClientStats, EvalServerStats

K_MAX = int(MAX_LEGAL_ACTIONS_PY)
TOKEN_DIM = int(TokenDataSize.TOKEN_DIM)


class SharedEvalBuffers:
    """Pre-allocated shared memory for zero-copy worker <-> server communication.

    All tensors are created once in the main process via ``share_memory_()``
    and picked up by worker/server processes after fork/spawn. Workers write
    into their own row of each per-worker tensor; the server reads those rows,
    runs inference, and writes already-softmaxed sparse priors + canonical
    values back into the same per-worker slot.

    Inputs (worker → server):
        states     (W, B, num_tokens, token_dim) float32
        phase_ids  (W, B)                        int8
        action_ids (W, B, K_MAX)                 int16 (uint16 via bit-reinterp)
        n_legals   (W, B)                        int16

    Outputs (server → worker):
        priors     (W, B, K_MAX)           float32  (sparse, softmaxed on GPU)
        values     (W, B, num_players)     float32  (canonical order)

    ``MAX_ACTION_SIZE`` (the dense model-head width) intentionally does not
    appear in any shape here — the server gathers dense logits down to the
    sparse legal list before the copy-back. Shared-mem output footprint
    drops from ~46 MB (workers × batch × 14977 × 4) to ~0.8 MB.
    """

    def __init__(
        self,
        num_workers: int,
        batch_size: int,
        num_players: int,
    ) -> None:
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_players = num_players
        self.num_tokens = get_num_tokens(num_players)
        self.token_dim = TOKEN_DIM
        self.k_max = K_MAX

        # --- Inputs (worker → server) ---
        self._states = torch.zeros(
            num_workers, batch_size, self.num_tokens, self.token_dim,
            dtype=torch.float32,
        ).share_memory_()
        self._phase_ids = torch.zeros(
            num_workers, batch_size, dtype=torch.int8,
        ).share_memory_()
        # torch.uint16 doesn't exist; store signed and reinterpret via
        # numpy .view(np.uint16). Max action id is 14976 ≪ 32767, so the
        # signed view is lossless and bit-identical.
        self._action_ids = torch.zeros(
            num_workers, batch_size, self.k_max, dtype=torch.int16,
        ).share_memory_()
        self._n_legals = torch.zeros(
            num_workers, batch_size, dtype=torch.int16,
        ).share_memory_()

        # --- Outputs (server → worker) ---
        # Sparse priors over the legal action list, already softmaxed on the
        # GPU inside the server's autocast region. Padded slots [n_legal:K_MAX]
        # carry near-zero mass (softmax of -1e9) and must never be read.
        self._priors = torch.zeros(
            num_workers, batch_size, self.k_max, dtype=torch.float32,
        ).share_memory_()
        self._values = torch.zeros(
            num_workers, batch_size, num_players, dtype=torch.float32,
        ).share_memory_()

        # Per-worker state counts (how many states in current request).
        self._counts = torch.zeros(num_workers, dtype=torch.int32).share_memory_()

        # Bitmap signaling and events — initialized lazily via init_bitmap().
        self._submitted_masks: torch.Tensor | None = None
        self._worker_to_server: np.ndarray | None = None
        self._worker_to_local_idx: np.ndarray | None = None
        self.server_events: list[Any] | None = None
        self.done_events: list[Any] | None = None

    # ------------------------------------------------------------------
    # Per-worker input accessors. Numpy views are created on-demand
    # because they don't survive pickling across spawn boundaries.
    # ------------------------------------------------------------------

    def get_input_states_np(self, worker_idx: int) -> np.ndarray:
        """(batch, num_tokens, token_dim) float32 view into this worker's input slot."""
        return self._states[worker_idx].numpy()

    def get_input_phase_ids_np(self, worker_idx: int) -> np.ndarray:
        """(batch,) int8 view into this worker's phase-id slot."""
        return self._phase_ids[worker_idx].numpy()

    def get_input_action_ids_np(self, worker_idx: int) -> np.ndarray:
        """(batch, K_MAX) uint16 view into this worker's action-id slot.

        The underlying storage is int16 because torch lacks uint16; this
        ``.view(np.uint16)`` is a bit-reinterpretation over the same memory.
        """
        return self._action_ids[worker_idx].numpy().view(np.uint16)

    def get_input_n_legals_np(self, worker_idx: int) -> np.ndarray:
        """(batch,) int16 view into this worker's n_legal slot."""
        return self._n_legals[worker_idx].numpy()

    # ------------------------------------------------------------------
    # Per-worker output accessors.
    # ------------------------------------------------------------------

    def get_output_priors_np(self, worker_idx: int) -> np.ndarray:
        """(batch, K_MAX) float32 view of this worker's sparse-prior output slot."""
        return self._priors[worker_idx].numpy()

    def get_output_values_np(self, worker_idx: int) -> np.ndarray:
        """(batch, num_players) float32 view of this worker's value output slot."""
        return self._values[worker_idx].numpy()

    def init_bitmap(
        self,
        partitions: list[tuple[int, int]],
        mp_context: Any = None,
    ) -> None:
        """Initialize bitmap signaling and per-worker/server events.

        Must be called before passing SharedEvalBuffers to server/worker
        processes.

        Args:
            partitions: List of (worker_start, worker_end) per eval server.
            mp_context: Multiprocessing context for Event creation.
        """
        import multiprocessing
        ctx = mp_context or multiprocessing
        num_servers = len(partitions)

        # Per-server submitted bitmask (uint64). Use int64 torch tensor
        # (same bit width) because torch lacks uint64; reinterpreted on
        # the Cython side via .view(np.uint64).
        self._submitted_masks = torch.zeros(
            num_servers, dtype=torch.int64,
        ).share_memory_()

        # Worker -> server and worker -> local_idx mappings
        w2s = np.zeros(self.num_workers, dtype=np.int32)
        w2l = np.zeros(self.num_workers, dtype=np.int32)
        for server_id, (ws, we) in enumerate(partitions):
            for w in range(ws, we):
                w2s[w] = server_id
                w2l[w] = w - ws
        self._worker_to_server = w2s
        self._worker_to_local_idx = w2l

        # Per-server doorbell events (wake idle servers on new work)
        self.server_events = [ctx.Event() for _ in range(num_servers)]
        # Per-worker done events (wake workers after inference completes)
        self.done_events = [ctx.Event() for _ in range(self.num_workers)]

    def get_bitmap_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (submitted_masks_uint64, counts_int32) numpy views.

        Creates views on-demand (numpy views don't survive spawn pickling).
        """
        assert self._submitted_masks is not None
        return self._submitted_masks.numpy().view(np.uint64), self._counts.numpy()

    def get_worker_mapping(self, worker_idx: int) -> tuple[int, int]:
        """Return (server_id, local_idx) for a given worker."""
        assert self._worker_to_server is not None
        assert self._worker_to_local_idx is not None
        return int(self._worker_to_server[worker_idx]), int(self._worker_to_local_idx[worker_idx])


def _eval_server_main(
    model: torch.nn.Module,
    device: torch.device,
    shared_bufs: SharedEvalBuffers,
    stop_event: Any,
    ready_event: Any,
    stats_report_event: Any,
    stats_queue: Any,
    *,
    server_id: int,
    worker_start: int,
    worker_end: int,
    profile: bool,
    no_compile: bool,
    compile_kwargs: dict[str, Any] | None = None,
    gpu_vendor: str = "cpu",
    fixed_batch_workers: int | None = None,
    epoch_ending_flag: Any = None,
    eval_dtype: str | None = None,
) -> None:
    """Eval server process entry point.

    Runs batched GPU inference in a loop, scanning shared-memory flags for
    requests from its assigned worker partition [worker_start, worker_end).
    Each process has its own GIL and CUDA default stream, so multiple
    servers truly overlap.
    """
    try:
        _eval_server_serve(
            model, device, shared_bufs,
            stop_event, ready_event, stats_report_event, stats_queue,
            server_id=server_id,
            worker_start=worker_start, worker_end=worker_end,
            profile=profile, no_compile=no_compile,
            compile_kwargs=compile_kwargs,
            gpu_vendor=gpu_vendor,
            fixed_batch_workers=fixed_batch_workers,
            epoch_ending_flag=epoch_ending_flag,
            eval_dtype=eval_dtype,
        )
    except Exception:
        import traceback
        traceback.print_exc()


def _eval_server_serve(
    model: torch.nn.Module,
    device: torch.device,
    shared_bufs: SharedEvalBuffers,
    stop_event: Any,
    ready_event: Any,
    stats_report_event: Any,
    stats_queue: Any,
    *,
    server_id: int,
    worker_start: int,
    worker_end: int,
    profile: bool,
    no_compile: bool,
    compile_kwargs: dict[str, Any] | None = None,
    gpu_vendor: str = "cpu",
    fixed_batch_workers: int | None = None,
    epoch_ending_flag: Any = None,
    eval_dtype: str | None = None,
) -> None:
    """Inner serve loop for an eval server process.

    Two modes based on fixed_batch_workers:

    **Fixed-batch mode** (fixed_batch_workers is an int): Accumulates
    drained workers until target_workers are ready, then submits a
    consistent-size GPU batch. At end-of-epoch (epoch_ending_flag set),
    flushes partial batches with zero-padding (if compiled) to maintain
    a single compiled graph size.

    **Greedy mode** (fixed_batch_workers is None): Drains all pending
    requests and submits immediately with exact batch size. No padding,
    no accumulation.
    """
    # Prevent OpenMP oversubscription (same as worker processes)
    torch.set_num_threads(1)

    model.eval()

    use_cuda = device.type == "cuda"

    # Apply vendor-specific per-process GPU settings (TF32 for NVIDIA, etc.).
    if gpu_vendor != "cpu":
        from train.gpu import GpuConfig
        GpuConfig(vendor=gpu_vendor).apply_optimizations()

    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    eval_autocast_dtype: torch.dtype | None = _dtype_map.get(eval_dtype) if eval_dtype else None

    num_tokens = shared_bufs.num_tokens
    token_dim = shared_bufs.token_dim
    k_max = shared_bufs.k_max
    npl = shared_bufs.num_players

    # Optionally compile the model (per-process compilation).
    if not no_compile and use_cuda:
        ckw = compile_kwargs if compile_kwargs is not None else {}
        model = torch.compile(model, **ckw)  # type: ignore[assignment]
        model.eval()
        # reduce-overhead / cudagraphs benefit most when the warmup batch size
        # matches the steady-state inference shape. In fixed-batch mode we pad
        # to a constant GPU batch size, so warm up with that size here.
        warmup_n = (
            fixed_batch_workers * shared_bufs.batch_size
            if fixed_batch_workers is not None
            else 1
        )
        with torch.inference_mode(), torch.autocast(
            device.type,
            dtype=eval_autocast_dtype,
            enabled=eval_autocast_dtype is not None,
        ):
            dummy_s = torch.randn(
                warmup_n, num_tokens, token_dim, device=device,
            )
            dummy_p = torch.zeros(warmup_n, dtype=torch.int8, device=device)
            dummy_a = torch.zeros(warmup_n, k_max, dtype=torch.int16, device=device)
            dummy_nl = torch.zeros(warmup_n, dtype=torch.int16, device=device)
            model(dummy_s, dummy_p, dummy_a, dummy_nl)
            del dummy_s, dummy_p, dummy_a, dummy_nl
        torch.cuda.synchronize()

    # Allocate pinned CPU buffers and GPU tensors in this process's CUDA context
    partition_size = worker_end - worker_start
    max_batch = partition_size * shared_bufs.batch_size

    # --- Inputs: pinned CPU + GPU side ---
    pin_s = torch.empty(
        max_batch, num_tokens, token_dim,
        dtype=torch.float32, pin_memory=use_cuda,
    )
    pin_s_np = pin_s.numpy()
    # Flat 2-D view for the row-agnostic Cython memcpy gather.
    pin_s_flat_np = pin_s_np.reshape(max_batch, num_tokens * token_dim)
    gpu_s = torch.empty(
        max_batch, num_tokens, token_dim, dtype=torch.float32, device=device,
    )

    # GPU phase/action/n_legals buffers are zero-initialized once and the
    # tail [total_n:max_batch] is never written again — per-batch H→D copies
    # only touch [:total_n]. The model gathers [:padded_n] each forward,
    # so any padded-tail rows read in-range zeros for action_ids/phase_ids
    # and n_legals=0 (which masks the row out via the model's invalid mask).
    # This drops three per-batch .zero_() launches from the hot path.
    pin_phase_ids = torch.empty(max_batch, dtype=torch.int8, pin_memory=use_cuda)
    pin_phase_ids_np = pin_phase_ids.numpy()
    gpu_phase_ids = torch.zeros(max_batch, dtype=torch.int8, device=device)

    pin_action_ids = torch.empty(
        max_batch, k_max, dtype=torch.int16, pin_memory=use_cuda,
    )
    pin_action_ids_np = pin_action_ids.numpy()
    gpu_action_ids = torch.zeros(
        max_batch, k_max, dtype=torch.int16, device=device,
    )

    pin_n_legals = torch.empty(max_batch, dtype=torch.int16, pin_memory=use_cuda)
    pin_n_legals_np = pin_n_legals.numpy()
    gpu_n_legals = torch.zeros(max_batch, dtype=torch.int16, device=device)

    # --- Outputs: pinned CPU + GPU side ---
    # Priors are (max_batch, K_MAX) f32 (sparse, already softmaxed on GPU).
    # This REPLACES the old (max_batch, MAX_ACTION_SIZE=14977) logits buffer.
    pin_priors = torch.empty(
        max_batch, k_max, dtype=torch.float32, pin_memory=use_cuda,
    )
    pin_priors_np = pin_priors.numpy()
    gpu_priors = torch.empty(
        max_batch, k_max, dtype=torch.float32, device=device,
    )

    pin_val = torch.empty(max_batch, npl, dtype=torch.float32, pin_memory=use_cuda)
    pin_val_np = pin_val.numpy()
    gpu_val = torch.empty(max_batch, npl, dtype=torch.float32, device=device)

    stats: EvalServerStats | None = EvalServerStats() if profile else None
    _tp = _ti = 0.0

    # Cython gather/scatter helpers (nogil memcpy, no per-worker Python loop).
    from mcts.mcts_core import (
        gather_action_ids as _gather_action_ids,
        gather_n_legals as _gather_n_legals,
        gather_phase_ids as _gather_phase_ids,
        gather_states as _gather_states,
        scatter_results as _scatter_results,
        server_drain_bitmap as _drain,
        server_peek_bitmap as _peek,
    )
    # Contiguous views of the shared-memory tensors (views don't survive
    # pickling, so they're built per-process here).
    # states: (W, B, num_tokens, token_dim) f32 → flatten last two for gather.
    all_states_np = shared_bufs._states.numpy().reshape(
        shared_bufs.num_workers, shared_bufs.batch_size, num_tokens * token_dim,
    )
    all_phase_ids_np = shared_bufs._phase_ids.numpy()
    all_action_ids_np = shared_bufs._action_ids.numpy()
    all_n_legals_np = shared_bufs._n_legals.numpy()
    all_priors_np = shared_bufs._priors.numpy()
    all_values_np = shared_bufs._values.numpy()
    priors_row_bytes = k_max * 4  # f32 = 4 bytes per element

    # Bitmap and event handles
    submitted_masks, sig_counts = shared_bufs.get_bitmap_arrays()
    assert shared_bufs.done_events is not None
    assert shared_bufs.server_events is not None
    done_events = shared_bufs.done_events
    server_event = shared_bufs.server_events[server_id]

    # Pre-allocate request arrays (filled each drain, avoids per-batch alloc)
    _widx_buf = np.empty(partition_size, dtype=np.int32)
    _cnt_buf = np.empty(partition_size, dtype=np.int32)

    # --- Shared helpers (closures over setup state) ---

    def _infer_and_scatter(widx: np.ndarray, cnts: np.ndarray,
                           n_req: int, padded_n: int = 0) -> int:
        """Gather inputs, run GPU inference + gather+softmax, scatter sparse priors.

        Args:
            widx: Worker indices array (length n_req).
            cnts: Per-worker state counts (length n_req).
            n_req: Number of workers in this batch.
            padded_n: GPU batch size (0 = use actual total_n, no padding).

        Returns:
            total_n: Actual number of states processed.
        """
        nonlocal _ti
        # Gather all four inputs into the pinned buffers via Cython memcpy.
        total_n = _gather_states(
            pin_s_flat_np, all_states_np, widx, cnts, n_req,
        )
        _gather_phase_ids(
            pin_phase_ids_np, all_phase_ids_np, widx, cnts, n_req,
        )
        _gather_action_ids(
            pin_action_ids_np, all_action_ids_np, widx, cnts, n_req,
        )
        _gather_n_legals(
            pin_n_legals_np, all_n_legals_np, widx, cnts, n_req,
        )
        if padded_n == 0:
            padded_n = total_n

        if stats is not None:
            _ti = perf_counter()

        gpu_s_batch = gpu_s[:padded_n]
        gpu_s_batch[:total_n].copy_(pin_s[:total_n], non_blocking=True)
        gpu_phase_ids[:total_n].copy_(pin_phase_ids[:total_n], non_blocking=True)
        gpu_action_ids[:total_n].copy_(pin_action_ids[:total_n], non_blocking=True)
        gpu_n_legals[:total_n].copy_(pin_n_legals[:total_n], non_blocking=True)
        # Padded tail [total_n:padded_n] is permanently zero (see allocation).

        # Build per-phase row indices on host so the model's policy gather
        # can use index_select / index_copy_ instead of boolean masking
        # (which would force a per-iteration H←D sync — that scatter
        # dominated analyze_game profiles before this change). Padded
        # rows [total_n:padded_n] have phase_id=0 on the GPU side
        # (zero-init tail), so we mirror that on the host before reading.
        if padded_n > total_n:
            pin_phase_ids_np[total_n:padded_n] = 0
        phase_view = pin_phase_ids_np[:padded_n]
        phase_indices: list[torch.Tensor] = []
        for _p in range(NUM_PHASES):
            _idx_np = np.nonzero(phase_view == _p)[0].astype(np.int64, copy=False)
            _t = torch.from_numpy(_idx_np)
            if use_cuda:
                _t = _t.to(device, non_blocking=True)
            phase_indices.append(_t)

        with torch.inference_mode():
            # Autocast region is already active (entered once for the whole
            # serve loop). The model gathers per-row legal slices internally
            # and returns (padded_n, K_MAX) sparse logits, so softmaxing
            # inside the autocast region keeps the narrow logits tensor in
            # bf16/fp16 until the final f32 copy.
            logits_sparse, values = model(
                gpu_s_batch, gpu_phase_ids[:padded_n],
                gpu_action_ids[:padded_n], gpu_n_legals[:padded_n],
                phase_indices,
            )
            # logits_sparse: (padded_n, K_MAX)       in autocast dtype
            # values:        (padded_n, num_players) in autocast dtype
            gpu_priors[:total_n] = logits_sparse[:total_n].softmax(dim=1).to(torch.float32)
            gpu_val[:total_n] = values[:total_n].to(torch.float32)

            pin_priors[:total_n].copy_(gpu_priors[:total_n], non_blocking=True)
            pin_val[:total_n].copy_(gpu_val[:total_n], non_blocking=True)
            if use_cuda:
                torch.cuda.synchronize()

        if stats is not None:
            stats.record_batch(total_n, perf_counter() - _ti)

        _scatter_results(
            pin_priors_np.view(np.int8), pin_val_np,
            all_priors_np.view(np.int8), all_values_np,
            widx, cnts, n_req, priors_row_bytes,
        )
        for i in range(n_req):
            done_events[widx[i]].set()
        return total_n

    def _check_stats_report() -> None:
        """Send stats to main process if requested."""
        if stats_report_event is not None and stats_report_event.is_set():
            stats_queue.put(copy.copy(stats))
            if stats is not None:
                stats.reset()
            stats_report_event.clear()

    def _idle_wait() -> None:
        """Lost-wakeup-safe doorbell wait when no work is pending."""
        server_event.clear()
        if _peek(submitted_masks, server_id):
            return
        server_event.wait(timeout=0.1)

    # Signal ready only after all initialization (imports, buffer allocation,
    # event validation) has succeeded. This ensures workers won't be spawned
    # against a server that's about to die from e.g. a missing Cython rebuild.
    ready_event.set()

    # Persistent autocast context — entered once for the entire serve loop
    # rather than per-batch, to avoid accumulated context-manager state from
    # thousands of enter/exit cycles.
    #
    # cache_enabled=False is critical: the autocast cache stores bf16 weight
    # casts keyed by TensorImpl* and only clears on __exit__ (which never
    # happens here).  Since the main process updates fp32 weights in-place
    # via CUDA IPC (optimizer.step()), a stale cache makes weight updates
    # invisible to the eval server — the model would use epoch-0 weights
    # for the entire training run.
    _autocast_ctx = (
        torch.autocast(
            device.type,
            dtype=eval_autocast_dtype,
            enabled=eval_autocast_dtype is not None,
            cache_enabled=False,
        )
        if use_cuda else None
    )
    if _autocast_ctx is not None:
        _autocast_ctx.__enter__()

    if fixed_batch_workers is not None:
        # --- Fixed-batch accumulation loop ---
        target_workers = fixed_batch_workers
        fixed_batch_size = target_workers * shared_bufs.batch_size
        _accum_widx = np.empty(partition_size, dtype=np.int32)
        _accum_cnts = np.empty(partition_size, dtype=np.int32)
        _accum_n = 0

        while not stop_event.is_set():
            if stats is not None:
                _tp = perf_counter()

            # Drain bitmap into accumulation buffer.
            num_drained = _drain(
                submitted_masks, sig_counts,
                _widx_buf, _cnt_buf, server_id, worker_start,
            )
            if num_drained > 0:
                _accum_widx[_accum_n:_accum_n + num_drained] = _widx_buf[:num_drained]
                _accum_cnts[_accum_n:_accum_n + num_drained] = _cnt_buf[:num_drained]
                _accum_n += num_drained

            # Submit full batches from accumulator.
            while _accum_n >= target_workers:
                _infer_and_scatter(
                    _accum_widx[:target_workers],
                    _accum_cnts[:target_workers],
                    target_workers, fixed_batch_size,
                )
                remaining = _accum_n - target_workers
                if remaining > 0:
                    _accum_widx[:remaining] = _accum_widx[target_workers:_accum_n]
                    _accum_cnts[:remaining] = _accum_cnts[target_workers:_accum_n]
                _accum_n = remaining

            # Flush partial batch at end of epoch.
            if (
                _accum_n > 0
                and epoch_ending_flag is not None
                and epoch_ending_flag.value
            ):
                padded = fixed_batch_size if not no_compile else 0
                _infer_and_scatter(
                    _accum_widx[:_accum_n],
                    _accum_cnts[:_accum_n],
                    _accum_n, padded,
                )
                _accum_n = 0

            # Idle / doorbell handling.
            if _accum_n == 0:
                if num_drained == 0 and stats is not None:
                    stats.record_idle(perf_counter() - _tp)
                _check_stats_report()
                _idle_wait()
            else:
                # Have partial work — brief wait for more arrivals.
                _check_stats_report()
                server_event.clear()
                if not _peek(submitted_masks, server_id):
                    server_event.wait(timeout=0.01)
    else:
        # --- Greedy loop (no fixed batch, no padding) ---
        while not stop_event.is_set():
            if stats is not None:
                _tp = perf_counter()

            num_requests = _drain(
                submitted_masks, sig_counts,
                _widx_buf, _cnt_buf, server_id, worker_start,
            )

            if num_requests == 0:
                if stats is not None:
                    stats.record_idle(perf_counter() - _tp)
                _check_stats_report()
                _idle_wait()
                continue

            _infer_and_scatter(
                _widx_buf[:num_requests], _cnt_buf[:num_requests],
                num_requests,
            )
            _check_stats_report()

    if _autocast_ctx is not None:
        _autocast_ctx.__exit__(None, None, None)


class EvaluationServer:
    """Process-based centralized NN evaluator using shared memory.

    Each server owns a contiguous partition of workers [worker_start,
    worker_end) and claims pending requests via atomic bitmap exchange.
    A per-server doorbell event wakes idle servers; per-worker done
    events signal completion to workers.

    Each EvaluationServer runs in its own process with a separate GIL,
    eliminating GIL contention between servers. The model's GPU parameter
    tensors are shared via CUDA IPC (torch.multiprocessing), so weight
    updates from the trainer are visible automatically.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        shared_bufs: SharedEvalBuffers,
        *,
        server_id: int = 0,
        worker_start: int = 0,
        worker_end: int | None = None,
        profile: bool = False,
        mp_context: Any = None,
        no_compile: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
        gpu_vendor: str = "cpu",
        fixed_batch_workers: int | None = None,
        epoch_ending_flag: Any = None,
        eval_dtype: str | None = None,
    ) -> None:
        import multiprocessing
        ctx = mp_context or multiprocessing
        if worker_end is None:
            worker_end = shared_bufs.num_workers
        self._stop_event = ctx.Event()
        self._ready_event = ctx.Event()
        self._stats_report_event: Any = ctx.Event() if profile else None
        self._stats_queue: Any = ctx.Queue() if profile else None
        self._process: Any | None = None
        self._process_args = (
            model, device, shared_bufs,
            self._stop_event, self._ready_event,
            self._stats_report_event, self._stats_queue,
        )
        self._process_kwargs = {
            "server_id": server_id,
            "worker_start": worker_start,
            "worker_end": worker_end,
            "profile": profile,
            "no_compile": no_compile,
            "compile_kwargs": compile_kwargs,
            "gpu_vendor": gpu_vendor,
            "fixed_batch_workers": fixed_batch_workers,
            "epoch_ending_flag": epoch_ending_flag,
            "eval_dtype": eval_dtype,
        }
        self._mp_context = ctx
        self._server_id = server_id

    def start(self) -> None:
        """Start the server process."""
        self._stop_event.clear()
        p = self._mp_context.Process(
            target=_eval_server_main,
            args=self._process_args,
            kwargs=self._process_kwargs,
            daemon=True,
        )
        p.start()
        self._process = p

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Block until the server has finished compilation and warmup.

        Returns True if the server signaled ready, False on timeout.
        """
        return self._ready_event.wait(timeout=timeout)

    def stop(self) -> None:
        """Signal the server to stop and wait for it."""
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    def get_profile_stats(self) -> EvalServerStats | None:
        """Request stats from the server process and return them.

        Signals the server to report, then reads from the shared queue.
        Returns None if profiling is disabled or the server doesn't respond.
        """
        if self._stats_report_event is None:
            return None
        self._stats_report_event.set()
        try:
            return self._stats_queue.get(timeout=2.0)
        except _queue.Empty:
            return None

    def reset_profile_stats(self) -> None:
        """Reset profile stats for a new epoch.

        Stats are reset inside the server after reporting.
        Drain any leftover stats from previous collection.
        """
        if self._stats_queue is not None:
            while not self._stats_queue.empty():
                try:
                    self._stats_queue.get_nowait()
                except _queue.Empty:
                    break


class RemoteEvaluator(BaseEvaluator):
    """Worker-side proxy that evaluates states via shared memory + EvaluationServer.

    Implements the same evaluate / evaluate_leaves / evaluate_terminal
    interface as NNEvaluator, so it can be used as a drop-in replacement.

    Hot path is pure numpy: workers fill ``(num_tokens, token_dim)`` f32
    token buffers via ``core.token_data.get_token_data``, write phase_id /
    action_ids / n_legal scalars into their shared-mem slots, publish a
    bitmap bit, and sleep on an Event until the server signals completion.
    The returned priors are already softmaxed over the legal action list
    on the GPU — no worker-side gather or softmax.

    Communication uses per-server uint64 bitmaps for lockfree request
    submission (atomic fetch-or) and per-worker mp.Events for done
    notification.

    Invariant: each worker may have at most one outstanding eval request
    at a time. Output slots are keyed by worker_idx alone (no request id),
    so a second request before the first completes would overwrite the
    output buffer. The sequential submit → wait → read flow enforces this.
    """

    # Seconds to wait for eval server response before raising.
    _EVAL_TIMEOUT = 30.0

    def __init__(
        self,
        num_players: int,
        shared_bufs: SharedEvalBuffers,
        worker_idx: int,
        *,
        profile: bool = False,
        terminal_rank_weight: float = 0.5,
    ) -> None:
        super().__init__(num_players, terminal_rank_weight)
        self._worker_idx = worker_idx

        # Input views (worker writes these).
        self._in_states_np = shared_bufs.get_input_states_np(worker_idx)
        self._in_phase_ids_np = shared_bufs.get_input_phase_ids_np(worker_idx)
        self._in_action_ids_np = shared_bufs.get_input_action_ids_np(worker_idx)
        self._in_n_legals_np = shared_bufs.get_input_n_legals_np(worker_idx)

        # Output views (worker reads these; already softmaxed + canonical).
        self._out_priors_np = shared_bufs.get_output_priors_np(worker_idx)
        self._out_values_np = shared_bufs.get_output_values_np(worker_idx)

        self._profile = profile
        self._stats: EvalClientStats | None = EvalClientStats() if profile else None

        # Bitmap signaling: atomic fetch-or to publish, mp.Event for done.
        from mcts.mcts_core import worker_publish_request
        self._submitted_masks, self._sig_counts = shared_bufs.get_bitmap_arrays()
        self._publish = worker_publish_request
        self._server_id, self._local_idx = shared_bufs.get_worker_mapping(worker_idx)
        assert shared_bufs.done_events is not None, (
            "SharedEvalBuffers.init_bitmap() must be called before creating RemoteEvaluator"
        )
        assert shared_bufs.server_events is not None
        self._done_event = shared_bufs.done_events[worker_idx]
        self._server_event = shared_bufs.server_events[self._server_id]

    def _request_eval(self, n: int) -> None:
        """Publish request via bitmap, wake server if needed, sleep until done."""
        self._done_event.clear()
        became_nonempty = self._publish(
            self._submitted_masks, self._sig_counts,
            self._worker_idx, self._server_id, self._local_idx, n,
        )
        if became_nonempty:
            self._server_event.set()
        if not self._done_event.wait(timeout=self._EVAL_TIMEOUT):
            raise RuntimeError(
                f"Eval server did not respond within {self._EVAL_TIMEOUT}s "
                f"(worker {self._worker_idx}, batch {n})"
            )

    def evaluate(
        self, state: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Evaluate a single game state via the remote server.

        Writes the token buffer / phase_id / legal action ids straight into
        the worker's shared-mem input slot, then blocks on the server's done
        event. Returns sparse softmax priors + canonical values (no
        worker-side gather or softmax).

        Returns:
            (priors_legal, values_canonical, action_ids_legal, n_legal, phase_id):
            - priors_legal: (n_legal,) float32 softmax over legal actions.
            - values_canonical: (num_players,) float32 in canonical order.
            - action_ids_legal: (n_legal,) uint16 phase-local legal action ids.
            - n_legal: count of legal actions.
            - phase_id: decision phase id 0-7.
        """
        phase_id = get_decision_phase_py(state)
        get_token_data(state, self._in_states_np[0])
        self._in_phase_ids_np[0] = phase_id
        # enumerate writes phase-local action ids directly into shared mem.
        n = enumerate_legal_actions_py(state, self._in_action_ids_np[0])
        self._in_n_legals_np[0] = n

        self._request_eval(1)

        return (
            self._out_priors_np[0, :n].copy(),
            self._out_values_np[0].copy(),
            self._in_action_ids_np[0, :n].copy(),
            n,
            phase_id,
        )

    def evaluate_leaves(
        self,
        state_arrays: list[np.ndarray],
        phase_ids: list[int],
        action_ids_buf: np.ndarray,
        n_legals: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Evaluate pre-enumerated leaf data in a single round-trip to the server.

        MCTS has already enumerated legal actions + computed phase ids during
        selection, so we just fill the per-leaf input rows and submit one
        batch. Returned priors are already softmaxed over each leaf's legal
        list on the GPU.

        Args:
            state_arrays: Raw int16 state arrays (pool row views), one per
                leaf. Each ``(total_size,)`` int16.
            phase_ids: Decision phase ids per leaf, length n.
            action_ids_buf: Legal phase-local action ids, shape
                ``(>= n, K_MAX)`` uint16. Only ``[i, :n_legals[i]]`` is read.
            n_legals: Count of legal actions per leaf, length n.

        Returns:
            List of ``(priors_legal[:n], values_canonical)`` tuples.
        """
        # Deferred import: keep GameState out of top-level to avoid a circular
        # import when core.state depends on Cython modules at init time.
        from core.state import GameState

        n = len(state_arrays)
        if n == 0:
            return []

        _stats = self._stats
        _t0 = _t1 = _t2 = 0.0
        if _stats is not None:
            _t0 = perf_counter()

        # Rebind a scratch GameState across rows — get_token_data needs a
        # GameState wrapper, but we don't want to allocate a new one per leaf.
        scratch_gs = GameState.from_buffer(state_arrays[0], self.num_players)
        get_token_data(scratch_gs, self._in_states_np[0])
        self._in_phase_ids_np[0] = phase_ids[0]
        n0 = n_legals[0]
        self._in_action_ids_np[0, :n0] = action_ids_buf[0, :n0]
        self._in_n_legals_np[0] = n0

        for i in range(1, n):
            scratch_gs.rebind(state_arrays[i], self.num_players)
            get_token_data(scratch_gs, self._in_states_np[i])
            self._in_phase_ids_np[i] = phase_ids[i]
            ni = n_legals[i]
            self._in_action_ids_np[i, :ni] = action_ids_buf[i, :ni]
            self._in_n_legals_np[i] = ni

        if _stats is not None:
            _t1 = perf_counter()
            _stats.prepare_secs += _t1 - _t0

        self._request_eval(n)

        if _stats is not None:
            _t2 = perf_counter()
            _stats.wait_secs += _t2 - _t1

        results = [
            (
                self._out_priors_np[i, :n_legals[i]].copy(),
                self._out_values_np[i].copy(),
            )
            for i in range(n)
        ]

        if _stats is not None:
            _stats.result_secs += perf_counter() - _t2
            _stats.num_calls += 1
            _stats.total_states += n

        return results

    def reset_profile_stats(self) -> None:
        """Reset profile stats for a new game."""
        if self._profile:
            self._stats = EvalClientStats()

    def get_profile_stats(self) -> EvalClientStats | None:
        """Return accumulated profile stats (None if profiling disabled)."""
        return self._stats
