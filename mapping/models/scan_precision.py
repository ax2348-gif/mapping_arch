"""``scan_precision``: the arithmetic of the Gated DeltaNet scan.

``tf32`` (the default; every model before this option): fla's fused
chunked kernel (``chunk_gated_delta_rule``, chunk 64) on the inputs as
they come (bf16 under the trainer's autocast). Its float32 matmuls run in
TF32: Triton's default for fp32 ``tl.dot``, and fla 0.5.2's fused chunk-64
WY solve hard-codes TF32 (``SOLVE_TRIL_DOT_PRECISION`` in
``fla/ops/gated_delta_rule/chunk_fwd.py``), which no environment setting
reaches. Against the exact recurrence this costs up to ~0.1 on outputs of
size ~1.5 in trained layers where beta is near 2 and the decay near 1
(cond(I + A) up to ~700; the error enters the carried state and does not
decay), 2-5% relative error on gradients, and up to 26% of held-out MSE
on se2c hybrid4c (experiments/gdn_kernel/, branch exp/gdn-kernel-check).

``ieee``: the same kernel family made exact to float32:
1. q, k, v, g, beta are cast to float32 into the kernel (also under
   autocast) and the output back to v's dtype;
2. chunk size 32, so the WY solve is fla's standalone ``solve_tril``,
   whose dots are IEEE (``FLA_TRIL_PRECISION`` defaults to ``ieee``),
   not the fused chunk-64 kernel;
3. every other fp32 ``tl.dot`` in IEEE: ``TRITON_F32_DEFAULT=ieee`` is set
   around each forward and backward kernel call (``triton_f32``) and
   restored after. Triton reads it when it compiles a kernel and it is in
   its on-disk cache key; its in-memory cache key is the specialization
   only, so the ieee path must never share a specialization with the tf32
   path. It cannot: every chunk kernel takes the chunk size as a constexpr
   (32 here, 64 in tf32 mode), and the solve here is a different kernel
   (tests/test_scan_precision.py runs both orders in one process).
Measured: forward within ~1e-5 of the float64 recurrence (the float32
recurrence's own error), gradients ~1e-6 relative. Cost: the fp32 scan is
about 2.7x the bf16 one (the scan only, not the step).

A model that has Gated DeltaNet layers declares ``SCAN_PRECISIONS`` in
its own class body and passes the setting to every ``GatedDeltaNet``
(``set_scan_precision``); ``ieee`` on any other class is refused, as an
undeclared ``query_mode`` is.
"""
import contextlib
import os

import torch

SCAN_PRECISIONS = ('tf32', 'ieee')
IEEE_CHUNK = 32


def check_scan_precision(model, precision):
    if precision not in SCAN_PRECISIONS:
        raise ValueError(f'scan_precision={precision!r}: must be one of {SCAN_PRECISIONS}')
    if precision != 'tf32' and precision not in type(model).__dict__.get('SCAN_PRECISIONS', ()):
        raise ValueError(f'{type(model).__name__} does not implement scan_precision='
                         f'{precision!r} (mapping/models/scan_precision.py)')


def set_scan_precision(model, precision, gdn_class):
    """Every ``gdn_class`` layer of ``model`` scans at ``precision``."""
    n = 0
    for m in model.modules():
        if isinstance(m, gdn_class):
            m.scan_precision = precision
            n += 1
    if precision != 'tf32' and n == 0:
        raise ValueError(f'scan_precision={precision!r}: {type(model).__name__} has no '
                         f'Gated DeltaNet layer')


@contextlib.contextmanager
def triton_f32(value):
    """``TRITON_F32_DEFAULT=value`` within, the previous value (or its
    absence) restored after."""
    old = os.environ.get('TRITON_F32_DEFAULT')
    os.environ['TRITON_F32_DEFAULT'] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ['TRITON_F32_DEFAULT']
        else:
            os.environ['TRITON_F32_DEFAULT'] = old


class _IeeeScan(torch.autograd.Function):
    """fla's chunk kernel (chunk ``IEEE_CHUNK``) on float32 inputs with
    IEEE dots in forward and backward. The kernel's own autograd graph is
    built inside ``forward`` and differentiated inside ``backward``, so the
    environment setting is in force for both."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        needs = ctx.needs_input_grad
        ins = [x.detach().float().contiguous().requires_grad_(n)
               for x, n in zip((q, k, v, g, beta), needs)]
        with torch.enable_grad(), torch.autocast('cuda', enabled=False), triton_f32('ieee'):
            o = chunk_gated_delta_rule(*ins, scale=1.0, chunk_size=IEEE_CHUNK)[0]
        ctx.ins, ctx.o = ins, o
        ctx.dtypes = [x.dtype for x in (q, k, v, g, beta)]
        return o.detach().to(v.dtype)

    @staticmethod
    def backward(ctx, do):
        wrt = [x for x in ctx.ins if x.requires_grad]
        with torch.autocast('cuda', enabled=False), triton_f32('ieee'):
            grads = iter(torch.autograd.grad(ctx.o, wrt, do.float()))
        out = [next(grads).to(dt) if x.requires_grad else None
               for x, dt in zip(ctx.ins, ctx.dtypes)]
        del ctx.ins, ctx.o
        return tuple(out)


def ieee_scan(q, k, v, g, beta):
    """``o`` (b, l, h, dv) in v's dtype: the gated delta rule (scale 1) on
    float32 inputs, chunk 32, IEEE dots."""
    return _IeeeScan.apply(q, k, v, g, beta)
