"""V3 full-graph capture, dynamic batches, and GPU compiler coverage.

CPU checks use AOTAutograd without GPU code generation. GPU checks use
the vendor's production Inductor options, fp16 wire inputs, and bf16 eval
autocast. The same eval tests run locally on ROCm and on NVIDIA in the cloud.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._dynamo.decorators import mark_unbacked
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters

from core.data import PHASE_ACTION_SIZES
from nn import _load_model_module
from nn.policy_layout import NUM_PHASES, UNIFIED_LOGIT_DIM, build_action_lut
from tests.test_transformer_v3_relations import _inputs
from train.gpu import detect_gpu


@pytest.fixture
def model():
    torch.manual_seed(42)
    module = _load_model_module('nn/transformer-v3.py')
    net = module.RSSTransformerNet(module.TransformerConfig(
        num_players=5, d_model=32, num_heads=4, num_layers=2,
    ))
    with torch.no_grad():
        net.relation_bias_mult.normal_(std=.1)
    yield net
    torch._dynamo.reset()


def _batch(model, size, *, sparse, device='cpu', wire=False):
    # Schema-valid synthetic inputs cover every policy block, varying player
    # capacity masks, nonempty quantity/ownership relations, and hidden keys.
    _, visible, dense, coords = _inputs(model, device)
    rows = torch.arange(size, device=device) % visible.shape[0]
    x = torch.randn(size, model.cfg.num_tokens, model.cfg.token_dim, device=device)
    x[:, :, 0] = visible[rows]
    x[:, model._player_slice, 1] = 0
    x[:, model._player_slice.start, 1] = 1
    x[:, model._corp_slice, 1] = 0
    x[:, model._corp_slice.start, 1] = 1
    x[:, model._company_slice, 1] = 0
    x[:, model._company_slice.start, 1] = 1
    legal = torch.zeros(size, UNIFIED_LOGIT_DIM, dtype=torch.bool, device=device)
    lut = build_action_lut().to(device)
    for row in range(size):
        phase = row % NUM_PHASES
        legal[row, lut[phase, :PHASE_ACTION_SIZES[phase]]] = True
    return x.half() if wire else x, legal, (coords if sparse else dense)[rows]


@pytest.mark.parametrize('sparse', [False, True])
def test_dynamic_fullgraph_capture_without_recompilation(model, sparse):
    """Input values, phase masks, and batch sizes including one share a graph."""
    model.eval()
    backend = CompileCounterWithBackend('aot_eager')
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    for size in (NUM_PHASES, 1, 7, 3, NUM_PHASES):
        batch = _batch(model, size, sparse=sparse)
        for tensor in batch:
            mark_unbacked(tensor, 0)
        # CPU flash-SDPA in torch 2.9 guards on unbacked batch == 1.
        # Use math SDPA here; GPU tests exercise the production kernels.
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            expected = model(*batch)
            actual = compiled(*batch)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)
    assert backend.frame_count == 1


def _loss(output, legal):
    policy, values = output
    targets = legal.float() / legal.sum(-1, keepdim=True)
    return -(targets * policy.log_softmax(-1)).sum(-1).mean() + values.square().mean()


def _assert_backward_parity(model, compiled, batch):
    model.zero_grad(set_to_none=True)
    expected = model(*batch)
    _loss(expected, batch[1]).backward()
    expected_grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    actual = compiled(*batch)
    _loss(actual, batch[1]).backward()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
    for name, parameter in model.named_parameters():
        if name in expected_grads:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected_grads[name], rtol=1e-3, atol=1e-5, msg=name)
        else:
            assert parameter.grad is None, name


def test_dense_fullgraph_training_backward(model):
    model.train()
    backend = CompileCounterWithBackend('aot_eager')
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    for _ in range(2):
        _assert_backward_parity(model, compiled, _batch(model, NUM_PHASES, sparse=False))
    assert backend.frame_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/ROCm GPU required')
@pytest.mark.parametrize('shape_mode', ['bucketed', 'dynamic'])
def test_gpu_sparse_eval_fullgraph(model, shape_mode):
    gpu = detect_gpu('cuda')
    gpu.apply_optimizations()
    model = model.cuda().eval()
    counters.clear()
    compiled = torch.compile(model, fullgraph=True, **gpu.get_compile_kwargs(
        for_training=False, eval_batch_shape_mode=shape_mode,
    ))
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 8) if shape_mode == 'bucketed' else (11, 1, 7, 32, 3)
    for size in sizes:
        batch = _batch(model, size, sparse=True, device='cuda', wire=True)
        if shape_mode == 'dynamic':
            for tensor in batch:
                mark_unbacked(tensor, 0)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            expected = model(*batch)
            actual = compiled(*batch)
            # Compare before the next CUDA-graph replay can overwrite outputs.
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, rtol=.03, atol=.003)
    assert not counters['graph_break']
    if shape_mode == 'dynamic':
        assert counters['stats']['unique_graphs'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/ROCm GPU required')
def test_gpu_dense_training_fullgraph(model):
    model = model.cuda().train()
    gpu = detect_gpu('cuda')
    gpu.apply_optimizations()
    compiled = torch.compile(model, fullgraph=True, **gpu.get_compile_kwargs(for_training=True))
    _assert_backward_parity(model, compiled, _batch(model, NUM_PHASES, sparse=False, device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/ROCm GPU required')
def test_sparse_eval_explicit_graph_replay_with_changed_edges_and_weights(model):
    """Fullgraph is insufficient: replay one captured graph with fresh data."""
    model = model.cuda().eval()
    gpu = detect_gpu('cuda')
    gpu.apply_optimizations()
    kwargs = gpu.get_compile_kwargs(for_training=False, eval_batch_shape_mode='bucketed')
    kwargs['options']['triton.cudagraphs'] = False  # Capture explicitly below.
    compiled = torch.compile(model, fullgraph=True, dynamic=False, **kwargs)
    batch = _batch(model, 8, sparse=True, device='cuda', wire=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(*batch)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = compiled(*batch)
        for empty in (False, True, False):
            changed = _batch(model, 8, sparse=True, device='cuda', wire=True)
            if empty:
                changed[2].zero_()
            for destination, source in zip(batch, changed):
                destination.copy_(source)
            # Weight snapshots must take effect without recapture or cached copies.
            model.relation_input_mixing.relation_gains.add_(.05)
            graph.replay()
            expected = compiled(*batch)
            for actual, reference in zip(output, expected):
                torch.testing.assert_close(actual, reference, rtol=.03, atol=.003)
