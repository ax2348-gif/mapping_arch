"""``query_mode``: how query tokens take part in a model's token mixers.

``mixed`` (the default; every model before this option): query tokens are
ordinary tokens of the sorted sequence. They write the Gated DeltaNet state
(in both scan directions), sit in the windows of the short convolution, and
attention is full over every token, so history tokens attend to queries and
queries to each other. The history representation, and so every answer,
then depends on how many queries there are and where.

``inert``: query tokens only read.
1. Gated DeltaNet: a query token has beta = 0 and log decay g = 0 in every
   head and both scan directions, so it reads the state without changing it.
2. Short convolution: history tokens are convolved over the history-only
   sequence (gathered, convolved in each head's scan order, scattered back),
   so their windows are exactly those of a sequence without queries. Query
   tokens get no convolution (identity; the SiLU after it is kept, as it is
   per token).
3. Attention: history -> history only; a query's tokens -> every history
   token and that query's own tokens; never another query's. RoPE positions
   are unchanged. A causal variant intersects this with its causal mask.
Everything else (MLPs, norms, readout) is per token. So in inert mode the
history tokens' activations at every layer do not depend on the queries,
and a query's prediction does not depend on the other queries
(tests/test_query_mode.py).

``QueryLayout`` carries what the mixers need, built once per forward from
the model's token permutation.

Masked attention, every mode: one path, FlexAttention (``attend``). A mask
is a ``mask_mod`` over per-token metadata (``QueryLayout.qid``: 0 for a
history token, j + 1 for query j) and token indices (causal: key index <=
query index in the sorted order), turned into a block-sparse ``BlockMask``
by ``create_block_mask``. Nothing in it is a Python constant of the query
count or T, and flex_attention and create_block_mask are compiled with
dynamic shapes, so a new query count or sequence length does not recompile
(tests/test_query_mode.py counts). Memory is linear in the tokens (no
L x L mask). Full bidirectional attention (mixed, non-causal) needs no mask
and stays plain SDPA. On CPU (tests) flex_attention runs eagerly.
"""
import functools

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

QUERY_MODES = ('mixed', 'inert')


def check_query_mode(model, query_mode):
    """Refuse an unknown mode, and ``inert`` on a class that does not
    declare it in its own body (``QUERY_MODES``): a subclass that changes
    the mixers must implement and declare inert itself, never inherit it."""
    if query_mode not in QUERY_MODES:
        raise ValueError(f'query_mode={query_mode!r}: must be one of {QUERY_MODES}')
    if query_mode != 'mixed' and query_mode not in type(model).__dict__.get('QUERY_MODES', ()):
        raise ValueError(f'{type(model).__name__} does not implement query_mode='
                         f'{query_mode!r} (mapping/models/query_mode.py)')


@functools.cache
def _compiled_flex():
    return torch.compile(flex_attention, dynamic=True)


def attend(q, k, v, block_mask):
    """Softmax attention of ``q, k, v (b, h, L, d)`` under ``block_mask``
    (scale 1/sqrt(d), as SDPA)."""
    if q.is_cuda:
        return _compiled_flex()(q, k, v, block_mask=block_mask)
    return flex_attention(q, k, v, block_mask=block_mask)


@functools.cache
def _compiled_block_mask():
    return torch.compile(create_block_mask, dynamic=True)


def block_mask(mask_mod, B, L, device):
    """``create_block_mask`` over ``L`` tokens (compiled on CUDA, so the
    mask is never materialised at L x L)."""
    device = torch.device(device)
    make = _compiled_block_mask() if device.type == 'cuda' else create_block_mask
    return make(mask_mod, B, None, L, L, device=device)


def mask_mod(qid, causal):
    """The one mask: token i may attend to token k when k is history
    (``qid == 0``) or i and k belong to the same query, and, if ``causal``
    (a 0-dim bool tensor), k <= i in sorted order. ``qid`` (b, L): 0 for
    history, j + 1 for query j. One code object for every mask, so the
    compiled kernels have one cache entry per head count and grad mode."""
    def mod(b, h, qi, ki):
        return (((qid[b, ki] == 0) | (qid[b, qi] == qid[b, ki]))
                & ((ki <= qi) | ~causal))
    return mod


@functools.cache
def _flag(causal, device):
    """A 0-dim bool tensor on ``device``, made once (no copy per forward)."""
    return torch.tensor(causal, device=device)


@functools.lru_cache(maxsize=64)
def causal_block_mask(b, L, device):
    """Causal over the sorted token order (mixed causal models: every token
    history)."""
    qid = torch.zeros(b, L, dtype=torch.long, device=device)
    return block_mask(mask_mod(qid, _flag(True, torch.device(device))), b, L, device)


class QueryLayout:
    """Where the query tokens sit in a sorted sequence.

    ``group`` (b, L) long, in sorted order: 0 for a history token
    (observation and odometry), j + 1 for a token of query j (also
    ``qid``, the metadata the attention mask reads).
    ``hist`` (b, L) bool: history tokens.
    ``hist_idx`` (b, n_hist) long: the sorted positions of the history
    tokens, increasing (the sort keeps history in layout order).
    """

    def __init__(self, order, n_hist, q, per_query):
        """``order`` (b, L): the model's permutation (sorted position s
        holds layout token ``order[:, s]``); the layout is ``n_hist``
        history tokens then ``q`` queries of ``per_query`` tokens each."""
        dev = order.device
        group = torch.cat([torch.zeros(n_hist, dtype=torch.long, device=dev),
                           torch.arange(1, q + 1, device=dev).repeat_interleave(per_query)])
        self.group = self.qid = group[order]                       # (b, L)
        self.hist = self.group == 0
        self.hist_idx = torch.argsort(order, 1)[:, :n_hist]        # (b, n_hist)
        self._masks = {}

    def mask_mod(self, causal=False):
        """``mask_mod`` over this layout's ``qid``."""
        return mask_mod(self.qid, _flag(bool(causal), self.qid.device))

    def block_mask(self, causal=False):
        """The ``BlockMask`` of ``mask_mod``, built once per forward."""
        if causal not in self._masks:
            b, L = self.qid.shape
            self._masks[causal] = block_mask(self.mask_mod(causal), b, L, self.qid.device)
        return self._masks[causal]

    def gather(self, x):
        """History tokens of ``x`` (b, L, ...) in sorted order."""
        idx = self.hist_idx.view(*self.hist_idx.shape, *([1] * (x.dim() - 2)))
        return torch.gather(x, 1, idx.expand(-1, -1, *x.shape[2:]))

    def scatter(self, x, hist):
        """``x`` with its history tokens replaced by ``hist``."""
        idx = self.hist_idx.view(*self.hist_idx.shape, *([1] * (x.dim() - 2)))
        return x.scatter(1, idx.expand(-1, -1, *x.shape[2:]), hist)
