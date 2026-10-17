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
"""
import torch

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


class QueryLayout:
    """Where the query tokens sit in a sorted sequence.

    ``group`` (b, L) long, in sorted order: 0 for a history token
    (observation and odometry), j + 1 for a token of query j.
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
        self.group = group[order]                                  # (b, L)
        self.hist = self.group == 0
        self.hist_idx = torch.argsort(order, 1)[:, :n_hist]        # (b, n_hist)
        self._masks = {}

    def attn_mask(self, causal=False):
        """(b, 1, L, L) bool, True where token i (row) may attend to token
        k (column): k is history, or i and k belong to the same query;
        ``causal`` also requires k <= i in sorted order."""
        if causal not in self._masks:
            g = self.group
            m = (g[:, None, :] == 0) | (g[:, :, None] == g[:, None, :])
            if causal:
                L = g.shape[1]
                m = m & torch.ones(L, L, dtype=torch.bool, device=g.device).tril()
            self._masks[causal] = m[:, None]
        return self._masks[causal]

    def gather(self, x):
        """History tokens of ``x`` (b, L, ...) in sorted order."""
        idx = self.hist_idx.view(*self.hist_idx.shape, *([1] * (x.dim() - 2)))
        return torch.gather(x, 1, idx.expand(-1, -1, *x.shape[2:]))

    def scatter(self, x, hist):
        """``x`` with its history tokens replaced by ``hist``."""
        idx = self.hist_idx.view(*self.hist_idx.shape, *([1] * (x.dim() - 2)))
        return x.scatter(1, idx.expand(-1, -1, *x.shape[2:]), hist)
