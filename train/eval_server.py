"""Centralized NN evaluation server for multi-process self-play.

Each EvaluationServer runs as a separate process with its own Python GIL,
eliminating GIL contention that limited throughput when servers were threads.
The model's GPU parameter tensors are shared zero-copy via CUDA IPC
(torch.multiprocessing), so optimizer.step() in the main process is visible
to eval servers automatically.

Communication uses shared memory (torch tensors with share_memory_()):

  Worker → Server (inputs)
      states      : transformer:
                    float16 (W, B, num_tokens, token_dim)
                    ResNet:
                    float16 (W, B, resnet_vector_dim)
      legal_mask  : uint8   (W, B, UNIFIED_LOGIT_DIM)  — 1=legal slot
      relation_coords (transformer only)
                 : uint8   (W, B, MAX_ATTENTION_RELATION_EDGES, 3)
                    — sparse (relation_id, query_token, key_token) triplets.
                    The model builds per-layer attention bias directly from
                    these sparse coordinates.

  Server → Worker (outputs — dense over unified slots, softmaxed on GPU)
      priors  : float32 (W, B, UNIFIED_LOGIT_DIM)  — legal slots sum to 1
      values  : float32 (W, B, num_players)
                transformer: canonical order
                ResNet: active-relative order on the wire; the worker
                unrotates before returning values to MCTS

Transformer token buffers are fp16 on the wire (half the shm + pinned + H→D
bytes of the old fp32 path). Cython's ``get_token_data`` still fills fp32, and
the worker casts into fp16 once at the shm boundary (one vectorized numpy
astype per eval request). Model-side the fp16 input is upcast to bf16 at
trunk entry, matching the autocast dtype the rest of the forward runs
under; all token feature values sit in [-1, ~3] post-normalization, so
fp16 round-trip error is <1e-3 absolute (see scratchpad/fp16_token_precision.py).
ResNet vector buffers use the same fp16 wire representation, then the server
upcasts them to fp32 before calling the ResNet.

The unified policy tensor (width ``UNIFIED_LOGIT_DIM``) is the wire format.
Workers build a dense legal-mask (via the same ``build_action_lut`` mapping
the model uses) from the phase-local action ids they already enumerated for
MCTS, then ship the mask rather than the sparse (phase_id, action_ids,
n_legals) triple. The server feeds the mask straight into the model's
forward, which masks-then-softmaxes on-device; the returned row is a proper
distribution over legal slots (illegal slots carry near-zero mass from
softmax of -1e9). Workers gather the per-leaf legal prior slice out of
that dense row using the same LUT, without any softmax on the worker side.

Wins vs the old sparse IPC protocol (1024-wide sparse action buffers):
  * inbound action_ids/n_legals/phase_ids collapses to a
    UNIFIED_LOGIT_DIM-wide uint8 mask.
  * outbound priors (1024 f32 = 4096 B/state) shrinks to
    UNIFIED_LOGIT_DIM f32 values.
  * on-GPU per-phase gather and n_legals scan drop out of the model
    forward — torch.compile sees a fully static (B, UNIFIED_LOGIT_DIM)
    policy path.

**Signaling protocol:**

Request submission is lockfree via per-server uint64 bitmap *arrays*
(Cython atomics in mcts_core.pyx). Each server owns ``W = ceil(max
partition size / 64)`` uint64 words, with each word padded to its own
64-byte cache line to avoid false sharing at partition sizes > 64.
Workers atomically set their bit in the appropriate word via fetch-or
(release); the server atomically exchanges each word to zero (acquire)
to claim all pending work in O(W + k ready). A per-server mp.Event
doorbell wakes idle servers. Done-notification is a per-server
mp.Condition plus a shared-memory uint8 ``done_flags`` array: the
scatter thread takes the condition lock once per batch, stamps
``done_flags[widx]=1`` for every completed worker, and calls
``notify_all`` a single time — eliminating the per-worker Event.set
burst that used to dominate the scatter thread.

**Multi-server concurrency:**

Each EvaluationServer owns a static partition of workers [worker_start,
worker_end) and only scans its partition's bitmap. Each server process
has its own GIL, CUDA context, and default stream, so multiple servers
truly overlap. Gather/scatter between per-worker slots and contiguous
inference buffers uses Cython nogil memcpy (no per-worker Python loop).

Workers do zero torch operations on the hot path — pure numpy writes
into the shared input buffers, a single bitmap publish + Event wait,
then numpy reads of the already-softmaxed sparse priors / values. ResNet
workers unrotate active-relative values to canonical order before handing
them to MCTS; transformer workers pass canonical values through unchanged.

RemoteEvaluator is the worker-side proxy that implements the same
``evaluate`` / ``evaluate_leaves`` / ``evaluate_terminal`` interface as
NNEvaluator, writing to shared memory instead of serializing over pipes.
"""

from __future__ import annotations

import copy
import queue as _queue
import signal
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch._dynamo.decorators import mark_unbacked

from core.actions import (
    enumerate_legal_actions_py,
    get_decision_phase_py,
)
from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
)
from core.data import MAX_ACTION_SIZE, PHASE_ACTION_SIZES
from core.relations import get_relation_coord_data, get_relation_coord_data_batch
from core.resnet_data import (
    get_resnet_data,
    get_resnet_data_batch,
    get_resnet_vector_size,
)
from core.token_data import (
    TokenDataSize,
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
)
from mcts.evaluator import BaseEvaluator
from nn.model_contract import (
    ModelInputSpec,
    ModelKind,
    normalize_model_type,
    unrotate_values_to_canonical,
)
from nn.transformer import NUM_PHASES, UNIFIED_LOGIT_DIM, build_action_lut
from train.profile_stats import EvalClientStats, EvalServerStats

TOKEN_DIM=int(TokenDataSize.TOKEN_DIM)


@dataclass(frozen=True)
class RequestBatchGroup:
    start: int
    end: int
    actual_n: int


def _next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def _resolve_actual_launch_cap(
    *,
    max_batch: int,
    batch_shape_mode: str,
    max_batch_size: int,
) -> int:
    if batch_shape_mode == "bucketed" and max_batch_size > 0:
        return max_batch_size
    return max_batch


def _resolve_max_launch_batch_size(
    *,
    max_batch: int,
    batch_shape_mode: str,
    max_batch_size: int,
) -> int:
    actual_launch_cap = _resolve_actual_launch_cap(
        max_batch=max_batch,
        batch_shape_mode=batch_shape_mode,
        max_batch_size=max_batch_size,
    )
    if batch_shape_mode == "bucketed":
        return _next_power_of_two(actual_launch_cap)
    return actual_launch_cap


def _resolve_launch_batch_size(*, actual_n: int, batch_shape_mode: str) -> int:
    if batch_shape_mode == "bucketed":
        return _next_power_of_two(actual_n)
    return actual_n


def _partition_request_groups(
    cnts: np.ndarray,
    *,
    n_req: int,
    actual_launch_cap: int,
) -> list[RequestBatchGroup]:
    if actual_launch_cap < 1:
        raise ValueError(
            f"actual_launch_cap must be >= 1, got {actual_launch_cap}"
        )
    groups: list[RequestBatchGroup] = []
    start = 0
    actual_n = 0
    for idx in range(n_req):
        cnt = int(cnts[idx])
        if cnt < 0:
            raise ValueError(f"request count must be >= 0, got {cnt}")
        if cnt > actual_launch_cap:
            raise ValueError(
                f"request count {cnt} exceeds actual_launch_cap {actual_launch_cap}"
            )
        if actual_n > 0 and actual_n + cnt > actual_launch_cap:
            groups.append(RequestBatchGroup(start=start, end=idx, actual_n=actual_n))
            start = idx
            actual_n = 0
        actual_n += cnt
    if actual_n > 0:
        groups.append(RequestBatchGroup(start=start, end=n_req, actual_n=actual_n))
    return groups


def _materialize_relation_coords_(
    *,
    dense_rel: torch.Tensor,
    dense_rel_flat: torch.Tensor,
    relation_coords: torch.Tensor,
    flat_idx: torch.Tensor,
    flat_idx_flat: torch.Tensor,
    tmp_idx: torch.Tensor,
    batch_offsets: torch.Tensor,
    sentinel_flat: torch.Tensor,
    actual_n: int,
    launch_n: int,
    num_tokens: int,
) -> None:
    """Materialize sparse relation coordinates into dense relation planes.

    ``relation_coords`` is uint8 ``(B, max_edges, 3)`` where each row is
    ``(relation_id, query_token, key_token)``. Padded rows are ``(0, 0, 0)``;
    after filling all coordinates, the per-batch sentinel slot is cleared so
    padding cannot introduce a real edge.
    """
    dense_rel[:launch_n].zero_()
    if actual_n <= 0:
        return

    coords = relation_coords[:actual_n]
    idx = flat_idx[:actual_n]
    tmp = tmp_idx[:actual_n]

    relation_stride = num_tokens * num_tokens
    idx.copy_(coords[..., 0])
    idx.mul_(relation_stride)
    tmp.copy_(coords[..., 1])
    idx.add_(tmp, alpha=num_tokens)
    tmp.copy_(coords[..., 2])
    idx.add_(tmp)
    idx.add_(batch_offsets[:actual_n])

    dense_rel_flat.index_fill_(0, flat_idx_flat[:idx.numel()], 1)
    dense_rel_flat.index_fill_(0, sentinel_flat[:actual_n], 0)


class SharedEvalBuffers:
    """Pre-allocated shared memory for zero-copy worker <-> server communication.

    All tensors are created once in the main process via ``share_memory_()``
    and picked up by worker/server processes after fork/spawn. Workers write
    into their own row of each per-worker tensor; the server reads those rows,
    runs inference, and writes already-softmaxed dense priors + model value
    outputs back into the same per-worker slot.

    Inputs (worker → server):
        states      transformer:
                    (W, B, num_tokens, token_dim)       float16
                    ResNet:
                    (W, B, resnet_vector_dim)           float16
        legal_mask  (W, B, UNIFIED_LOGIT_DIM)           uint8 (1=legal slot)
        relation_coords (transformer only)
                    (W, B, MAX_ATTENTION_RELATION_EDGES, 3)
                    uint8 sparse relation triplets

    Outputs (server → worker):
        priors  (W, B, UNIFIED_LOGIT_DIM)  float32  (softmaxed over legal slots)
        values  (W, B, num_players)        float32
                transformer: canonical order
                ResNet: active-relative order until RemoteEvaluator unrotates

    The dense unified representation carries legality via a per-slot mask
    — no sparse action-id buffer appears in any shape here. Shared-mem
    output footprint is ``workers × batch × UNIFIED_LOGIT_DIM × 4`` bytes.
    """

    def __init__(
        self,
        num_workers: int,
        batch_size: int,
        num_players: int,
        input_spec: ModelInputSpec | None = None,
    ) -> None:
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_players = num_players
        self.unified_dim = UNIFIED_LOGIT_DIM
        if input_spec is None:
            input_spec = ModelInputSpec(
                model_type=ModelKind.TRANSFORMER.value,
                num_players=num_players,
                policy_dim=int(UNIFIED_LOGIT_DIM),
                value_dim=num_players,
                input_dim=None,
                num_tokens=get_num_tokens(num_players),
                token_dim=TOKEN_DIM,
                uses_relations=True,
                values_are_active_relative=False,
            )
        model_kind = normalize_model_type(input_spec.model_type)
        if int(input_spec.num_players) != num_players:
            raise ValueError(
                f"input_spec num_players ({input_spec.num_players}) does not "
                f"match num_players ({num_players})"
            )
        if int(input_spec.policy_dim) != int(UNIFIED_LOGIT_DIM):
            raise ValueError(
                f"input_spec policy_dim ({input_spec.policy_dim}) does not "
                f"match UNIFIED_LOGIT_DIM ({int(UNIFIED_LOGIT_DIM)})"
            )
        if int(input_spec.value_dim) != num_players:
            raise ValueError(
                f"input_spec value_dim ({input_spec.value_dim}) does not "
                f"match num_players ({num_players})"
            )
        self.model_type = model_kind.value
        self.uses_relations = bool(input_spec.uses_relations)
        self.values_are_active_relative = bool(
            input_spec.values_are_active_relative
        )
        self.num_tokens = int(input_spec.num_tokens or 0)
        self.token_dim = int(input_spec.token_dim or 0)
        self.input_dim = int(input_spec.input_dim or 0)

        if model_kind is ModelKind.TRANSFORMER:
            expected_tokens = get_num_tokens(num_players)
            if self.num_tokens != expected_tokens:
                raise ValueError(
                    f"Transformer num_tokens ({self.num_tokens}) does not "
                    f"match get_num_tokens({num_players}) ({expected_tokens})"
                )
            if self.token_dim != TOKEN_DIM:
                raise ValueError(
                    f"Transformer token_dim ({self.token_dim}) does not "
                    f"match TOKEN_DIM ({TOKEN_DIM})"
                )
            if self.input_dim != 0:
                raise ValueError("Transformer input_dim must be unset")
            if not self.uses_relations:
                raise ValueError("Transformer shared buffers require relations")
            if self.values_are_active_relative:
                raise ValueError("Transformer values must be canonical")
        else:
            expected_input_dim = get_resnet_vector_size(num_players)
            if self.input_dim != expected_input_dim:
                raise ValueError(
                    f"ResNet input_dim ({self.input_dim}) does not match "
                    f"get_resnet_vector_size({num_players}) "
                    f"({expected_input_dim})"
                )
            if self.num_tokens != 0 or self.token_dim != 0:
                raise ValueError("ResNet token dimensions must be unset")
            if self.uses_relations:
                raise ValueError("ResNet shared buffers do not use relations")
            if not self.values_are_active_relative:
                raise ValueError("ResNet values must be active-relative")

        self.num_relations = (
            NUM_ATTENTION_RELATIONS if self.uses_relations else 0
        )
        self.max_relation_edges = (
            MAX_ATTENTION_RELATION_EDGES if self.uses_relations else 0
        )
        self.relation_coord_width = (
            ATTENTION_RELATION_COORD_WIDTH if self.uses_relations else 0
        )

        # --- Inputs (worker → server) ---
        if model_kind is ModelKind.TRANSFORMER:
            # fp16 on the wire: halves shm footprint + pinned→GPU copy vs.
            # the old fp32 buffer. Cython's ``get_token_data`` still writes
            # fp32 into a per-worker scratch; the worker casts into this fp16
            # slot at the shm boundary.
            self._states = torch.zeros(
                num_workers, batch_size, self.num_tokens, self.token_dim,
                dtype=torch.float16,
            ).share_memory_()
        else:
            self._states = torch.zeros(
                num_workers, batch_size, self.input_dim, dtype=torch.float16,
            ).share_memory_()
        self._legal_mask = torch.zeros(
            num_workers, batch_size, self.unified_dim, dtype=torch.uint8,
        ).share_memory_()
        if self.uses_relations:
            # Sparse Graphormer-style directed relation coordinates. The eval
            # model consumes these directly and builds dense SDPA bias tensors
            # per layer, avoiding dense relation planes on the IPC/server path.
            self._relation_coords: torch.Tensor | None = torch.zeros(
                num_workers, batch_size,
                self.max_relation_edges, self.relation_coord_width,
                dtype=torch.uint8,
            ).share_memory_()
        else:
            self._relation_coords = None

        # --- Outputs (server → worker) ---
        # Dense priors over unified slots, already masked + softmaxed on the
        # GPU inside the server's autocast region. Illegal slots carry near-
        # zero mass (softmax of -1e9). Callers use the same ``action_lut``
        # that built the mask to gather the per-leaf legal prior slice.
        self._priors = torch.zeros(
            num_workers, batch_size, self.unified_dim, dtype=torch.float32,
        ).share_memory_()
        self._values = torch.zeros(
            num_workers, batch_size, num_players, dtype=torch.float32,
        ).share_memory_()

        # Per-worker state counts (how many states in current request).
        self._counts = torch.zeros(num_workers, dtype=torch.int32).share_memory_()

        # Per-worker done flags (server → worker). Written by the scatter
        # thread under its server's ``done_cond`` lock; read by the owning
        # worker both under the lock (for the wait-loop predicate) and
        # outside (for the pre-publish clear, which is safe because the
        # flag is only ever transitioned 1→0 by its owner between request
        # cycles and 0→1 by the server after the bitmap submission).
        self._done_flags = torch.zeros(num_workers, dtype=torch.uint8).share_memory_()

        # Bitmap signaling and events — initialized lazily via init_bitmap().
        self._submitted_masks: torch.Tensor | None = None
        self._num_words: int = 0
        self._worker_to_server: np.ndarray | None = None
        self._worker_to_local_idx: np.ndarray | None = None
        self.server_events: list[Any] | None = None
        self.done_conds: list[Any] | None = None

    # ------------------------------------------------------------------
    # Per-worker input accessors. Numpy views are created on-demand
    # because they don't survive pickling across spawn boundaries.
    # ------------------------------------------------------------------

    def get_input_states_np(self, worker_idx: int) -> np.ndarray:
        """View into this worker's model-input slot.

        Transformer shape is ``(batch, num_tokens, token_dim)`` float16.
        ResNet shape is ``(batch, resnet_vector_dim)`` float16.
        """
        return self._states[worker_idx].numpy()

    def get_input_vectors_np(self, worker_idx: int) -> np.ndarray:
        """(batch, resnet_vector_dim) float16 view into a ResNet input slot."""
        if self.model_type != ModelKind.RESNET.value:
            raise RuntimeError("input vectors are only allocated for ResNet")
        return self._states[worker_idx].numpy()

    def get_input_legal_mask_np(self, worker_idx: int) -> np.ndarray:
        """(batch, UNIFIED_LOGIT_DIM) uint8 view into this worker's legal-mask slot."""
        return self._legal_mask[worker_idx].numpy()

    def get_input_relation_coords_np(self, worker_idx: int) -> np.ndarray:
        """(batch, max_relation_edges, 3) uint8 sparse relation coordinates."""
        if self._relation_coords is None:
            raise RuntimeError(
                "relation coordinates are only allocated for transformer"
            )
        return self._relation_coords[worker_idx].numpy()

    # ------------------------------------------------------------------
    # Per-worker output accessors.
    # ------------------------------------------------------------------

    def get_output_priors_np(self, worker_idx: int) -> np.ndarray:
        """(batch, UNIFIED_LOGIT_DIM) float32 view of this worker's prior output slot."""
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

        # Per-server submitted bitmap: W = ceil(max_partition / 64) uint64
        # words per server, each word padded to its own 64-byte cache line
        # (column 0 is the live word; columns 1..7 are dead padding). Without
        # the padding, multiple (server, word) pairs share a cache line and
        # every publish triggers cross-partition RFO traffic that scales with
        # total worker count. Layout: ``(num_servers * W, 8)``; word ``w`` of
        # server ``s`` lives at row ``s*W + w``.
        #
        # Stored as int64 because torch lacks uint64; reinterpreted on the
        # Cython side via ``.view(np.uint64)``.
        max_partition = max(we - ws for ws, we in partitions)
        W = (max_partition + 63) // 64
        self._num_words = W
        self._submitted_masks = torch.zeros(
            num_servers * W, 8, dtype=torch.int64,
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
        # Per-server done-conditions (one shared condvar per server partition).
        # Workers in partition P wait on done_conds[P]; the server's scatter
        # thread notifies it exactly once per batch after stamping flags.
        self.done_conds = [ctx.Condition() for _ in range(num_servers)]

    def get_bitmap_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (submitted_masks_uint64, counts_int32) numpy views.

        Creates views on-demand (numpy views don't survive spawn pickling).
        ``submitted_masks`` is 2-D ``(num_servers * W, 8)`` uint64; the
        Cython helpers take the 2-D memoryview directly. Callers thread
        ``self._num_words`` through to the Cython publish/drain/peek calls.
        """
        assert self._submitted_masks is not None
        return self._submitted_masks.numpy().view(np.uint64), self._counts.numpy()

    def get_worker_mapping(self, worker_idx: int) -> tuple[int, int]:
        """Return (server_id, local_idx) for a given worker."""
        assert self._worker_to_server is not None
        assert self._worker_to_local_idx is not None
        return int(self._worker_to_server[worker_idx]), int(self._worker_to_local_idx[worker_idx])

    def get_done_flags_np(self) -> np.ndarray:
        """uint8 (num_workers,) view of the shared done-flag array.

        Numpy views over shared-memory torch tensors don't survive spawn
        pickling, so each process must call this for its own view.
        """
        return self._done_flags.numpy()


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
    min_batch_size: int = 0,
    min_batch_timeout_ms: float = 10.0,
    batch_shape_mode: str = "dynamic",
    max_batch_size: int = 0,
    eval_dtype: str | None = None,
) -> None:
    """Eval server process entry point.

    Runs batched GPU inference in a loop, scanning shared-memory flags for
    requests from its assigned worker partition [worker_start, worker_end).
    Each process has its own GIL and CUDA default stream, so multiple
    servers truly overlap.
    """
    # Main drives shutdown via stop_event; Ctrl-C SIGINT would otherwise
    # interrupt GPU inference or the scatter thread's Condition notify and
    # leave the worker-side lock in an inconsistent state.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _eval_server_serve(
            model, device, shared_bufs,
            stop_event, ready_event, stats_report_event, stats_queue,
            server_id=server_id,
            worker_start=worker_start, worker_end=worker_end,
            profile=profile, no_compile=no_compile,
            compile_kwargs=compile_kwargs,
            gpu_vendor=gpu_vendor,
            min_batch_size=min_batch_size,
            min_batch_timeout_ms=min_batch_timeout_ms,
            batch_shape_mode=batch_shape_mode,
            max_batch_size=max_batch_size,
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
    min_batch_size: int = 0,
    min_batch_timeout_ms: float = 10.0,
    batch_shape_mode: str = "dynamic",
    max_batch_size: int = 0,
    eval_dtype: str | None = None,
) -> None:
    """Inner serve loop for an eval server process.

    Two scheduling modes based on min_batch_size:

    **Min-batch mode** (min_batch_size > 0): Accumulates drained requests
    until the sum of pending states reaches min_batch_size, then submits.
    If the accumulator is non-empty but below the floor, waits on the
    doorbell with ``min_batch_timeout_ms`` and flushes the partial batch
    on timeout (anti-starvation for epoch end / root eval). Any oversized
    drained work is partitioned into contiguous launch groups before
    gather/infer/scatter.

    **Greedy mode** (min_batch_size == 0): Drains all pending requests
    and submits immediately. Any oversized drain is still split into one
    or more contiguous launch groups before inference.
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

    partition_size = worker_end - worker_start
    max_batch = partition_size * shared_bufs.batch_size
    actual_launch_cap = _resolve_actual_launch_cap(
        max_batch=max_batch,
        batch_shape_mode=batch_shape_mode,
        max_batch_size=max_batch_size,
    )
    max_launch_batch = _resolve_max_launch_batch_size(
        max_batch=max_batch,
        batch_shape_mode=batch_shape_mode,
        max_batch_size=max_batch_size,
    )

    model_kind = normalize_model_type(shared_bufs.model_type)
    uses_resnet_vectors = model_kind is ModelKind.RESNET
    num_tokens = shared_bufs.num_tokens
    token_dim = shared_bufs.token_dim
    input_dim = shared_bufs.input_dim
    u_dim = shared_bufs.unified_dim
    npl = shared_bufs.num_players

    # Optionally compile the model (per-process compilation).
    if not no_compile and use_cuda:
        ckw = compile_kwargs if compile_kwargs is not None else {}
        model = torch.compile(model, **ckw)  # type: ignore[assignment]
        model.eval()
        lut = build_action_lut()
        if batch_shape_mode == "dynamic":
            # Warm up with the EXACT call signature the hot path uses —
            # (states, legal_mask, relations). Dynamic-shape strategy (see
            # ``train/gpu/{amd,nvidia}.py`` for why we dropped global
            # ``dynamic=True``): the batch dim gets ``mark_unbacked`` so the
            # single compiled graph handles all batch sizes 1..max_batch, and
            # in particular sizes 0 and 1 don't get framework-level specialized.
            warmup_batches = [NUM_PHASES]
        else:
            warmup_batches = []
            warmup_n = 1
            while warmup_n <= max_launch_batch:
                warmup_batches.append(warmup_n)
                warmup_n <<= 1
        with torch.inference_mode(), torch.autocast(
            device.type,
            dtype=eval_autocast_dtype,
            enabled=eval_autocast_dtype is not None,
            # Match the hot-path persistent context (see ``cache_enabled``
            # rationale below). Dynamo guards on the full autocast state,
            # including ``cache_enabled``; a mismatch between warmup and
            # hot path triggers a one-shot recompile on the first serve.
            cache_enabled=False,
        ):
            for warmup_n in warmup_batches:
                if uses_resnet_vectors:
                    dummy_s = torch.randn(
                        warmup_n, input_dim, device=device, dtype=torch.float32,
                    )
                    dummy_rel = None
                else:
                    dummy_s = torch.randn(
                        warmup_n, num_tokens, token_dim, device=device,
                        dtype=torch.float16,
                    )
                    dummy_rel = torch.zeros(
                        warmup_n, shared_bufs.max_relation_edges,
                        shared_bufs.relation_coord_width,
                        dtype=torch.uint8, device=device,
                    )
                # Build one legal mask per phase so every phase's slots get
                # exercised at least once during compilation.
                dummy_mask = torch.zeros(
                    warmup_n, u_dim, dtype=torch.bool, device=device,
                )
                for i in range(warmup_n):
                    phase_id = i % NUM_PHASES
                    n = PHASE_ACTION_SIZES[phase_id]
                    dummy_mask[i, lut[phase_id, :n].to(device)] = True
                if batch_shape_mode == "dynamic":
                    mark_tensors = (
                        (dummy_s, dummy_mask)
                        if uses_resnet_vectors
                        else (dummy_s, dummy_mask, dummy_rel)
                    )
                    for _t in mark_tensors:
                        mark_unbacked(_t, 0)
                if uses_resnet_vectors:
                    model(dummy_s, dummy_mask)
                else:
                    model(dummy_s, dummy_mask, dummy_rel)
                del dummy_s, dummy_mask, dummy_rel
        torch.cuda.synchronize()

    # Allocate pinned CPU buffers and GPU tensors in this process's CUDA context.
    alloc_batch = max_launch_batch

    # Ping-pong depth for pinned I/O buffers. While the GPU computes batch
    # N and the scatter thread reads slot N's pinned outputs, the main loop
    # writes slot N+1's pinned inputs. A single event per slot, recorded
    # after the D→H copies, covers both the input (H→D read of pin_s[slot])
    # and output (D→H write of pin_priors[slot]) halves.
    buf_depth = 2 if use_cuda else 1

    # --- Inputs: pinned CPU + GPU side ---
    if uses_resnet_vectors:
        input_shape = (alloc_batch, input_dim)
        input_flat_width = input_dim
    else:
        input_shape = (alloc_batch, num_tokens, token_dim)
        input_flat_width = num_tokens * token_dim
    input_dtype = torch.float16
    states_row_bytes = input_flat_width * 2  # fp16 == 2 bytes

    pin_s_list = [
        torch.empty(*input_shape, dtype=input_dtype, pin_memory=use_cuda)
        for _ in range(buf_depth)
    ]
    pin_s_np_list = [p.numpy() for p in pin_s_list]
    # Flat byte view for the dtype-agnostic Cython memcpy gather.
    pin_s_flat_bytes_list = [
        p.reshape(alloc_batch, input_flat_width).view(np.int8)
        for p in pin_s_np_list
    ]
    gpu_s = torch.empty(*input_shape, dtype=input_dtype, device=device)
    gpu_s_model = (
        torch.empty(alloc_batch, input_dim, dtype=torch.float32, device=device)
        if uses_resnet_vectors else None
    )

    if uses_resnet_vectors:
        relation_coord_row_bytes = 0
        pin_rel_coord_list: list[torch.Tensor] = []
        pin_rel_coord_np_list: list[np.ndarray] = []
        pin_rel_coord_flat_bytes_list: list[np.ndarray] = []
        gpu_rel_coord = None
    else:
        max_relation_edges = shared_bufs.max_relation_edges
        relation_coord_width = shared_bufs.relation_coord_width
        relation_coord_row_bytes = max_relation_edges * relation_coord_width
        pin_rel_coord_list = [
            torch.empty(
                alloc_batch, max_relation_edges, relation_coord_width,
                dtype=torch.uint8, pin_memory=use_cuda,
            )
            for _ in range(buf_depth)
        ]
        pin_rel_coord_np_list = [p.numpy() for p in pin_rel_coord_list]
        pin_rel_coord_flat_bytes_list = [
            p.reshape(alloc_batch, relation_coord_row_bytes).view(np.int8)
            for p in pin_rel_coord_np_list
        ]
        gpu_rel_coord = torch.empty(
            alloc_batch, max_relation_edges, relation_coord_width,
            dtype=torch.uint8, device=device,
        )
    # Shared-memory mask gather stays uint8, but bucketed launches use a
    # preallocated bool tensor at the model call site so the GPU-visible
    # invocation shape is static apart from the chosen bucket size.
    pin_mask_list = [
        torch.empty(alloc_batch, u_dim, dtype=torch.uint8, pin_memory=use_cuda)
        for _ in range(buf_depth)
    ]
    pin_mask_np_list = [p.numpy() for p in pin_mask_list]
    gpu_mask = torch.zeros(alloc_batch, u_dim, dtype=torch.uint8, device=device)
    gpu_mask_bool = torch.zeros(alloc_batch, u_dim, dtype=torch.bool, device=device)

    # --- Outputs: pinned CPU + GPU side ---
    pin_priors_list = [
        torch.empty(
            alloc_batch, u_dim, dtype=torch.float32, pin_memory=use_cuda,
        )
        for _ in range(buf_depth)
    ]
    pin_priors_np_list = [p.numpy() for p in pin_priors_list]
    gpu_priors = torch.empty(
        alloc_batch, u_dim, dtype=torch.float32, device=device,
    )

    pin_val_list = [
        torch.empty(alloc_batch, npl, dtype=torch.float32, pin_memory=use_cuda)
        for _ in range(buf_depth)
    ]
    pin_val_np_list = [p.numpy() for p in pin_val_list]
    gpu_val = torch.empty(alloc_batch, npl, dtype=torch.float32, device=device)

    stats: EvalServerStats | None = EvalServerStats() if profile else None
    _tp = 0.0

    # Cython gather/scatter helpers (nogil memcpy, no per-worker Python loop).
    from mcts.mcts_core import (
        gather_masks as _gather_masks,
        gather_states as _gather_states,
        scatter_results as _scatter_results,
        server_drain_bitmap as _drain,
        server_peek_bitmap as _peek,
    )
    # Contiguous views of the shared-memory tensors (views don't survive
    # pickling, so they're built per-process here).
    # states: transformer fp16 tokens or ResNet fp16 vectors → flatten input
    # dims and view as bytes so the Cython gather is dtype-agnostic (matches
    # scatter_results).
    all_states_bytes = shared_bufs._states.numpy().reshape(
        shared_bufs.num_workers, shared_bufs.batch_size, input_flat_width,
    ).view(np.int8)
    if uses_resnet_vectors:
        all_relation_coords_bytes = None
    else:
        assert shared_bufs._relation_coords is not None
        all_relation_coords_bytes = shared_bufs._relation_coords.numpy().reshape(
            shared_bufs.num_workers, shared_bufs.batch_size, relation_coord_row_bytes,
        ).view(np.int8)
    all_masks_np = shared_bufs._legal_mask.numpy()
    all_priors_np = shared_bufs._priors.numpy()
    all_values_np = shared_bufs._values.numpy()
    priors_row_bytes = u_dim * 4  # f32 = 4 bytes per element

    # Bitmap and event handles
    submitted_masks, sig_counts = shared_bufs.get_bitmap_arrays()
    num_words = shared_bufs._num_words
    assert shared_bufs.done_conds is not None
    assert shared_bufs.server_events is not None
    done_cond = shared_bufs.done_conds[server_id]
    done_flags_np = shared_bufs.get_done_flags_np()
    server_event = shared_bufs.server_events[server_id]

    # Pre-allocate request arrays (filled each drain, avoids per-batch alloc)
    _widx_buf = np.empty(partition_size, dtype=np.int32)
    _cnt_buf = np.empty(partition_size, dtype=np.int32)

    # --- Async-completion infrastructure ---------------------------------
    # The scatter thread takes the blocking cuda.synchronize() and the
    # done-notify work off the main serve loop's critical path. Profiling
    # before this change showed 70% of wall time in synchronize and 14%
    # in the per-worker Event.set burst — both are now pipelined behind
    # batch N+1's input prep and GPU launch, and the burst itself has
    # collapsed to a single notify_all on a shared condition.
    #
    # Slot ownership:
    #   main thread  — writes pin_*[slot] inputs, launches GPU work,
    #                  records a CUDA event after the D→H copies, hands
    #                  the slot off via _scatter_jobs.
    #   scatter thr  — event.synchronize() (releases GIL → main runs free),
    #                  reads pin_priors[slot]/pin_val[slot] into shared
    #                  memory, stamps per-worker done_flags and notifies
    #                  the done-condition, returns the slot via _free_slots.
    #
    # The single recorded event covers BOTH halves of the slot's lifetime:
    # the H→D copies must have completed (GPU copy engine finished reading
    # pin_s[slot]) before the model forward could run, and the D→H copies
    # must have completed before the event was visible. So by the time the
    # scatter thread wakes, pin_s[slot] is safe to overwrite AND
    # pin_priors[slot]/pin_val[slot] are fully populated.
    _free_slots: _queue.Queue[int] = _queue.Queue(maxsize=buf_depth)
    for _i in range(buf_depth):
        _free_slots.put(_i)
    # Sentinel `None` tells the scatter thread to exit.
    _scatter_jobs: _queue.Queue[Any] = _queue.Queue()
    # Serializes stats access between main (record_idle, copy+reset) and
    # scatter thread (record_batch).
    _stats_lock = threading.Lock()

    def _scatter_worker() -> None:
        while True:
            job = _scatter_jobs.get()
            if job is None:
                return
            (event, slot, widx_copy, cnts_copy, n_req, actual_n,
             start_ti) = job
            if event is not None:
                event.synchronize()
            _scatter_results(
                pin_priors_np_list[slot].view(np.int8), pin_val_np_list[slot],
                all_priors_np.view(np.int8), all_values_np,
                widx_copy, cnts_copy, n_req, priors_row_bytes,
            )
            # Consolidated wake: one lock acquire, N byte stores, one
            # notify_all. Replaces the old per-worker Event.set() burst
            # (N × (lock + notify + unlock)).
            with done_cond:
                for i in range(n_req):
                    done_flags_np[widx_copy[i]] = 1
                done_cond.notify_all()
            if stats is not None:
                with _stats_lock:
                    stats.record_batch(actual_n, perf_counter() - start_ti)
            _free_slots.put(slot)

    _scatter_thread = threading.Thread(
        target=_scatter_worker, name=f"eval-scatter-{server_id}", daemon=True,
    )
    _scatter_thread.start()

    # --- Shared helpers (closures over setup state) ---

    def _infer_and_scatter(widx: np.ndarray, cnts: np.ndarray,
                           n_req: int) -> int:
        """Gather inputs, launch GPU inference, hand off to scatter thread.

        Blocks on _free_slots.get() when all buf_depth slots are in flight;
        this is the backpressure that keeps the pipeline at depth
        ``buf_depth`` instead of letting the main loop run arbitrarily
        far ahead of the GPU.

        Args:
            widx: Worker indices array (length n_req). Copied before return.
            cnts: Per-worker state counts (length n_req). Copied before return.
            n_req: Number of workers in this batch.

        Returns:
            actual_n: Actual number of states processed.
        """
        # Acquire a free ping-pong slot. Blocks until the scatter thread
        # finishes processing an in-flight slot (the GPU sync happens
        # there, not here).
        slot = _free_slots.get()

        pin_s_slot = pin_s_list[slot]
        pin_s_flat_bytes = pin_s_flat_bytes_list[slot]
        if uses_resnet_vectors:
            pin_rel_coord_slot = None
            pin_rel_coord_flat_bytes = None
        else:
            pin_rel_coord_slot = pin_rel_coord_list[slot]
            pin_rel_coord_flat_bytes = pin_rel_coord_flat_bytes_list[slot]
        pin_mask_slot = pin_mask_list[slot]
        pin_mask_np = pin_mask_np_list[slot]
        pin_priors_slot = pin_priors_list[slot]
        pin_val_slot = pin_val_list[slot]

        # Gather model inputs + masks into the pinned buffers via Cython
        # memcpy. Inputs go through the byte-level path: transformer fp16
        # token rows or ResNet fp16 vector rows.
        actual_n = _gather_states(
            pin_s_flat_bytes, all_states_bytes, widx, cnts, n_req,
            states_row_bytes,
        )
        if not uses_resnet_vectors:
            assert pin_rel_coord_flat_bytes is not None
            assert all_relation_coords_bytes is not None
            relation_coord_n = _gather_states(
                pin_rel_coord_flat_bytes, all_relation_coords_bytes,
                widx, cnts, n_req, relation_coord_row_bytes,
            )
            if relation_coord_n != actual_n:
                raise AssertionError(
                    f"relation coord gather count {relation_coord_n} "
                    f"!= state gather count {actual_n}"
                )
        _gather_masks(pin_mask_np, all_masks_np, widx, cnts, n_req)
        launch_n = _resolve_launch_batch_size(
            actual_n=actual_n,
            batch_shape_mode=batch_shape_mode,
        )

        start_ti = perf_counter() if stats is not None else 0.0

        # Snapshot widx/cnts — the caller reuses these arrays for the next
        # drain, so the scatter thread must work from its own copy.
        widx_copy = widx[:n_req].copy()
        cnts_copy = cnts[:n_req].copy()

        gpu_s_batch = gpu_s[:launch_n]
        gpu_s_batch[:actual_n].copy_(pin_s_slot[:actual_n], non_blocking=True)
        if not uses_resnet_vectors:
            assert gpu_rel_coord is not None
            assert pin_rel_coord_slot is not None
            gpu_rel_coord[:actual_n].copy_(
                pin_rel_coord_slot[:actual_n],
                non_blocking=True,
            )
        gpu_mask[:actual_n].copy_(pin_mask_slot[:actual_n], non_blocking=True)
        if launch_n > actual_n:
            gpu_s_batch[actual_n:launch_n].zero_()
            if not uses_resnet_vectors:
                assert gpu_rel_coord is not None
                gpu_rel_coord[actual_n:launch_n].zero_()
            gpu_mask[actual_n:launch_n].zero_()
            gpu_mask_bool[actual_n:launch_n].zero_()

        if batch_shape_mode == "bucketed":
            gpu_mask_bool[:launch_n].copy_(gpu_mask[:launch_n], non_blocking=True)
            mask_batch = gpu_mask_bool[:launch_n]
        else:
            mask_batch = gpu_mask[:launch_n].to(torch.bool)

        with torch.inference_mode():
            # Autocast region is already active (entered once for the whole
            # serve loop). The model sees ``launch_n`` rows, but only the first
            # ``actual_n`` rows contain real gathered work and only those rows
            # are copied back to shared memory.
            if uses_resnet_vectors:
                assert gpu_s_model is not None
                gpu_s_model[:launch_n].copy_(gpu_s_batch, non_blocking=True)
                logits, values = model(gpu_s_model[:launch_n], mask_batch)
            else:
                assert gpu_rel_coord is not None
                logits, values = model(
                    gpu_s_batch, mask_batch, gpu_rel_coord[:launch_n],
                )
            gpu_priors[:actual_n] = (
                logits[:actual_n].softmax(dim=1).to(torch.float32)
            )
            gpu_val[:actual_n] = values[:actual_n].to(torch.float32)

            pin_priors_slot[:actual_n].copy_(gpu_priors[:actual_n], non_blocking=True)
            pin_val_slot[:actual_n].copy_(gpu_val[:actual_n], non_blocking=True)

        if use_cuda:
            event = torch.cuda.Event()
            event.record()
            _scatter_jobs.put((
                event, slot, widx_copy, cnts_copy, n_req, actual_n, start_ti,
            ))
        else:
            # CPU path: no event, scatter inline to keep semantics simple.
            _scatter_results(
                pin_priors_np_list[slot].view(np.int8), pin_val_np_list[slot],
                all_priors_np.view(np.int8), all_values_np,
                widx_copy, cnts_copy, n_req, priors_row_bytes,
            )
            with done_cond:
                for i in range(n_req):
                    done_flags_np[widx_copy[i]] = 1
                done_cond.notify_all()
            if stats is not None:
                stats.record_batch(actual_n, perf_counter() - start_ti)
            _free_slots.put(slot)
        return actual_n

    def _check_stats_report() -> None:
        """Send stats to main process if requested.

        Held under ``_stats_lock`` so the scatter thread's in-flight
        ``record_batch`` calls can't interleave a half-updated state into
        the snapshot or get lost across the reset.
        """
        if stats_report_event is not None and stats_report_event.is_set():
            with _stats_lock:
                snap = copy.copy(stats)
                if stats is not None:
                    stats.reset()
            stats_queue.put(snap)
            stats_report_event.clear()

    def _idle_wait() -> None:
        """Lost-wakeup-safe doorbell wait when no work is pending."""
        server_event.clear()
        if _peek(submitted_masks, server_id, num_words):
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

    if min_batch_size > 0:
        # --- Min-batch accumulation loop ---
        # Accumulator holds drained (widx, cnt) pairs whose total pending
        # state count is still below ``min_batch_size``. Units are states
        # throughout — no worker-level accounting — because the downstream
        # forward pass is sized in states and a worker's per-request state
        # count is a ceiling, not a hard value (locked MCTS frontier can
        # publish fewer than search_batch_size states).
        _accum_widx = np.empty(partition_size, dtype=np.int32)
        _accum_cnts = np.empty(partition_size, dtype=np.int32)
        _accum_n = 0          # number of worker entries in accumulator
        _accum_states = 0     # sum of cnts across accumulator entries
        _timeout_s = max(min_batch_timeout_ms, 0.0) / 1000.0

        while not stop_event.is_set():
            if stats is not None:
                _tp = perf_counter()

            # Drain bitmap into accumulation buffer.
            num_drained = _drain(
                submitted_masks, sig_counts,
                _widx_buf, _cnt_buf, server_id, worker_start, num_words,
            )
            if num_drained > 0:
                _accum_widx[_accum_n:_accum_n + num_drained] = _widx_buf[:num_drained]
                _accum_cnts[_accum_n:_accum_n + num_drained] = _cnt_buf[:num_drained]
                _accum_n += num_drained
                _accum_states += int(_cnt_buf[:num_drained].sum())

            # Submit once the accumulator crosses the floor. A single
            # drain can push us well past min_batch_size — submit all
            # of it in one forward pass (bounded by the partition's
            # natural max_batch).
            if _accum_states >= min_batch_size:
                for group in _partition_request_groups(
                    _accum_cnts,
                    n_req=_accum_n,
                    actual_launch_cap=actual_launch_cap,
                ):
                    _infer_and_scatter(
                        _accum_widx[group.start:group.end],
                        _accum_cnts[group.start:group.end],
                        group.end - group.start,
                    )
                _accum_n = 0
                _accum_states = 0

            # Idle / doorbell handling.
            if _accum_n == 0:
                if num_drained == 0 and stats is not None:
                    with _stats_lock:
                        stats.record_idle(perf_counter() - _tp)
                _check_stats_report()
                _idle_wait()
            else:
                # Have partial work — brief wait for more arrivals, then
                # flush whatever's accumulated on timeout (anti-starvation
                # for epoch end and single-worker root evals).
                _check_stats_report()
                server_event.clear()
                if not _peek(submitted_masks, server_id, num_words):
                    if not server_event.wait(timeout=_timeout_s):
                        for group in _partition_request_groups(
                            _accum_cnts,
                            n_req=_accum_n,
                            actual_launch_cap=actual_launch_cap,
                        ):
                            _infer_and_scatter(
                                _accum_widx[group.start:group.end],
                                _accum_cnts[group.start:group.end],
                                group.end - group.start,
                            )
                        _accum_n = 0
                        _accum_states = 0
    else:
        # --- Greedy loop (no min batch, no accumulation, no timeout) ---
        while not stop_event.is_set():
            if stats is not None:
                _tp = perf_counter()

            num_requests = _drain(
                submitted_masks, sig_counts,
                _widx_buf, _cnt_buf, server_id, worker_start, num_words,
            )

            if num_requests == 0:
                if stats is not None:
                    with _stats_lock:
                        stats.record_idle(perf_counter() - _tp)
                _check_stats_report()
                _idle_wait()
                continue

            for group in _partition_request_groups(
                _cnt_buf,
                n_req=num_requests,
                actual_launch_cap=actual_launch_cap,
            ):
                _infer_and_scatter(
                    _widx_buf[group.start:group.end],
                    _cnt_buf[group.start:group.end],
                    group.end - group.start,
                )
            _check_stats_report()

    # Drain the scatter thread: any batches launched before stop_event was
    # observed must finish their notify so workers aren't left stuck on
    # done_cond.wait(). The sentinel tells the thread to exit once it's
    # worked through the queue.
    _scatter_jobs.put(None)
    _scatter_thread.join()

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
        min_batch_size: int = 0,
        min_batch_timeout_ms: float = 10.0,
        batch_shape_mode: str = "dynamic",
        max_batch_size: int = 0,
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
            "min_batch_size": min_batch_size,
            "min_batch_timeout_ms": min_batch_timeout_ms,
            "batch_shape_mode": batch_shape_mode,
            "max_batch_size": max_batch_size,
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

    Hot path is pure numpy: workers fill transformer token buffers via
    ``core.token_data.get_token_data`` or ResNet vector buffers via
    ``core.resnet_data.get_resnet_data`` plus a dense ``UNIFIED_LOGIT_DIM``
    legal mask into their shared-mem slots, publish a bitmap bit, and sleep
    on the per-server done-condition until a shared-memory done flag flips.
    The returned priors are already masked + softmaxed on the GPU — no
    worker-side softmax.

    ResNet values are active-relative on the wire. The worker already has the
    source state rows, so it unrotates those values to canonical order before
    returning them to MCTS. Transformer values are canonical already.

    Communication uses per-server uint64 bitmap arrays (W cache-line-padded
    words each) for lockfree request submission (atomic fetch-or) and a
    per-server mp.Condition + shm ``done_flags`` array for done notification
    (one notify_all per batch instead of N per-worker Event.set() calls).

    Invariant: each worker may have at most one outstanding eval request
    at a time. Output slots are keyed by worker_idx alone (no request id),
    so a second request before the first completes would overwrite the
    output buffer. The sequential submit → wait → read flow enforces this.
    """

    # Seconds to wait for eval server response before raising.
    _EVAL_TIMEOUT = 1200.0

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
        self._model_kind = normalize_model_type(shared_bufs.model_type)
        self._uses_resnet_vectors = self._model_kind is ModelKind.RESNET
        self._values_are_active_relative = shared_bufs.values_are_active_relative

        # Input views (worker writes these).
        # Transformer ``_in_states_np`` is fp16 (wire dtype);
        # ``get_token_data`` and ``get_resnet_data`` write fp32, so we keep a
        # per-worker fp32 scratch and cast into the shm slot at the boundary.
        self._in_states_np = shared_bufs.get_input_states_np(worker_idx)
        self._in_legal_mask_np = shared_bufs.get_input_legal_mask_np(worker_idx)
        self._in_relation_coords_np = (
            shared_bufs.get_input_relation_coords_np(worker_idx)
            if shared_bufs.uses_relations else None
        )
        self._states_scratch_fp32 = np.empty(
            self._in_states_np.shape, dtype=np.float32,
        )
        self._active_players_np = np.empty(shared_bufs.batch_size, dtype=np.int16)

        # Output views (worker reads these; values may still need unrotation).
        self._out_priors_np = shared_bufs.get_output_priors_np(worker_idx)
        self._out_values_np = shared_bufs.get_output_values_np(worker_idx)

        # Worker-local LUT (phase_id, phase-local action id) → unified slot.
        # Used in the single-state ``evaluate`` path to set legal-mask bits
        # pre-publish and gather the sparse prior slice post-forward; the
        # dense ``evaluate_leaves`` path bypasses it entirely.
        self._action_lut_np: np.ndarray = build_action_lut().numpy()
        # Scratch for enumerate_legal_actions_py on the single-state path.
        self._enum_scratch: np.ndarray = np.empty(
            MAX_ACTION_SIZE, dtype=np.uint16,
        )

        self._profile = profile
        self._stats: EvalClientStats | None = EvalClientStats() if profile else None

        # Bitmap signaling: atomic fetch-or to publish, per-server
        # Condition + shm done flag for completion.
        from mcts.mcts_core import worker_publish_request
        self._submitted_masks, self._sig_counts = shared_bufs.get_bitmap_arrays()
        self._num_words = shared_bufs._num_words
        self._publish = worker_publish_request
        self._server_id, self._local_idx = shared_bufs.get_worker_mapping(worker_idx)
        assert shared_bufs.done_conds is not None, (
            "SharedEvalBuffers.init_bitmap() must be called before creating RemoteEvaluator"
        )
        assert shared_bufs.server_events is not None
        self._done_cond = shared_bufs.done_conds[self._server_id]
        self._done_flags_np = shared_bufs.get_done_flags_np()
        self._server_event = shared_bufs.server_events[self._server_id]

    def _unrotate_values_if_needed(
        self,
        values: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Return canonical values for the worker-facing evaluator contract."""
        if not self._values_are_active_relative:
            return values
        for i in range(n):
            values[i] = unrotate_values_to_canonical(
                values[i],
                int(self._active_players_np[i]),
                self.num_players,
            )
        return values

    def _fill_mask_row(
        self, i: int, phase_id: int, action_ids_legal: np.ndarray,
    ) -> None:
        """Zero shm mask row ``i``, then flip legal unified slots to 1.

        ``action_ids_legal`` is the phase-local id slice; the LUT maps
        each id into its slot in the ``UNIFIED_LOGIT_DIM``-wide mask.
        """
        self._in_legal_mask_np[i] = 0
        slots = self._action_lut_np[phase_id, action_ids_legal]
        self._in_legal_mask_np[i, slots] = 1

    def _request_eval(self, n: int) -> None:
        """Publish request via bitmap, wake server if needed, sleep until done.

        Uses a per-server mp.Condition + shared-memory ``done_flags``
        byte array. Clearing our flag outside the condition's lock is
        safe because this worker is the only actor that transitions it
        1→0 (and only here, strictly between response observation and
        next publish); the server only transitions it 0→1 after observing
        the bitmap bit we set in ``publish``.
        """
        self._done_flags_np[self._worker_idx] = 0
        became_nonempty = self._publish(
            self._submitted_masks, self._sig_counts,
            self._worker_idx, self._server_id, self._local_idx, n,
            self._num_words,
        )
        if became_nonempty:
            self._server_event.set()
        deadline = perf_counter() + self._EVAL_TIMEOUT
        with self._done_cond:
            while self._done_flags_np[self._worker_idx] == 0:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Eval server did not respond within {self._EVAL_TIMEOUT}s "
                        f"(worker {self._worker_idx}, batch {n})"
                    )
                self._done_cond.wait(timeout=remaining)

    def evaluate(
        self, state: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Evaluate a single game state via the remote server.

        Writes the model input / phase_id / legal action ids straight into
        the worker's shared-mem input slot, then blocks on the server's done
        event. Returns sparse softmax priors + canonical values.

        Returns:
            (priors_legal, values_canonical, action_ids_legal, n_legal, phase_id):
            - priors_legal: (n_legal,) float32 softmax over legal actions.
            - values_canonical: (num_players,) float32 in canonical order.
            - action_ids_legal: (n_legal,) uint16 phase-local legal action ids.
            - n_legal: count of legal actions.
            - phase_id: decision phase id 0-10.
        """
        actual_num_players = self._actual_num_players(state)
        phase_id = get_decision_phase_py(state)
        self._active_players_np[0] = int(
            state._array[self._active_player_state_offset]
        )
        if self._uses_resnet_vectors:
            get_resnet_data(state, self._states_scratch_fp32[0])
            self._in_states_np[0] = self._states_scratch_fp32[0].astype(np.float16)
        else:
            assert self._in_relation_coords_np is not None
            get_token_data(
                state, self._states_scratch_fp32[0],
                max_players=self.num_players,
            )
            self._in_states_np[0] = self._states_scratch_fp32[0].astype(np.float16)
            get_relation_coord_data(
                state, self._in_relation_coords_np[0],
                max_players=self.num_players,
            )
        n = enumerate_legal_actions_py(state, self._enum_scratch)
        self._fill_mask_row(0, phase_id, self._enum_scratch[:n])

        self._request_eval(1)

        slots = self._action_lut_np[phase_id, self._enum_scratch[:n]]
        values = self._out_values_np[:1].copy()
        self._unrotate_values_if_needed(values, 1)
        return (
            self._out_priors_np[0, slots].copy(),
            values[0, :actual_num_players].copy(),
            self._enum_scratch[:n].copy(),
            n,
            phase_id,
        )

    def evaluate_leaves(
        self,
        state_arrays: list[np.ndarray],
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate pre-masked leaf data in a single round-trip to the server.

        MCTS has already built the dense per-leaf legal mask during
        selection, so we copy model inputs + masks into shared memory and
        submit one batch. The server returns dense priors over unified slots;
        this method hands them back as a contiguous ``(n, UNIFIED_LOGIT_DIM)``
        view — the caller gathers the sparse legal slice (if needed) via
        whatever LUT they used to build the mask.

        Args:
            state_arrays: Raw int16 state arrays (pool row views), one per
                leaf. Each ``(total_size,)`` int16.
            legal_mask: Dense legal slot mask, shape
                ``(n, UNIFIED_LOGIT_DIM)`` uint8 or bool.

        Returns:
            ``(priors, values)`` where ``priors`` is
            ``(n, UNIFIED_LOGIT_DIM)`` float32 (softmaxed on the GPU with
            illegal slots near zero) and ``values`` is
            ``(n, num_players)`` float32 in canonical order. Both are
            fresh copies — the shared-mem slots are free after return.
        """
        n = len(state_arrays)
        if n == 0:
            return (
                np.empty((0, UNIFIED_LOGIT_DIM), dtype=np.float32),
                np.empty((0, self.num_players), dtype=np.float32),
            )
        actual_num_players = self._actual_num_players_for_rows(state_arrays)

        _stats = self._stats
        _t0 = _t1 = _t2 = 0.0
        if _stats is not None:
            _t0 = perf_counter()

        if self._uses_resnet_vectors:
            # ResNet vectors are active-relative; keep active-player ids
            # worker-local for post-read value unrotation.
            get_resnet_data_batch(
                state_arrays, self.num_players, self._states_scratch_fp32[:n],
            )
            self._in_states_np[:n] = self._states_scratch_fp32[:n].astype(
                np.float16
            )
            for i, state_array in enumerate(state_arrays):
                self._active_players_np[i] = int(
                    state_array[self._active_player_state_offset]
                )
        else:
            assert self._in_relation_coords_np is not None
            # Batched token fill — single Cython entry amortizes per-leaf
            # Python dispatch + GameState wrapper construction (rebind across
            # rows internally). Fill fp32 scratch first, then cast into the
            # fp16 shm slot at the boundary (one vectorized astype, ~tens of μs).
            get_token_data_batch(
                state_arrays, self._states_scratch_fp32[:n],
                max_players=self.num_players,
            )
            self._in_states_np[:n] = self._states_scratch_fp32[:n].astype(
                np.float16
            )
            get_relation_coord_data_batch(
                state_arrays, self._in_relation_coords_np[:n],
                max_players=self.num_players,
            )
        # Caller already has the dense mask — copy it straight into shm.
        np.copyto(self._in_legal_mask_np[:n], legal_mask, casting="unsafe")

        if _stats is not None:
            _t1 = perf_counter()
            _stats.prepare_secs += _t1 - _t0

        self._request_eval(n)

        if _stats is not None:
            _t2 = perf_counter()
            _stats.wait_secs += _t2 - _t1

        priors = self._out_priors_np[:n].copy()
        values = self._out_values_np[:n].copy()
        self._unrotate_values_if_needed(values, n)

        if _stats is not None:
            _stats.result_secs += perf_counter() - _t2
            _stats.num_calls += 1
            _stats.total_states += n

        return priors, values[:, :actual_num_players].copy()

    def reset_profile_stats(self) -> None:
        """Reset profile stats for a new game."""
        if self._profile:
            self._stats = EvalClientStats()

    def get_profile_stats(self) -> EvalClientStats | None:
        """Return accumulated profile stats (None if profiling disabled)."""
        return self._stats
