"""Discrete SE(2): poses with a heading, and the egocentric odometry
between them.

The walk is still sampled in the world frame (``mapping.data``). This
module turns a world pose sequence into what an agent would record:
observations rotated so that its heading is image-up, and one odometry
symbol per step giving the relative pose in the agent's own frame.

Conventions, all on ``(y, x)`` pixel vectors with y down:

- heading ``h`` in 0..3 means facing ``MOVES[h]``: 0 up, 1 right, 2 down,
  3 left.
- an observation is presented facing-up: ``np.rot90(window, h)``.
- a world vector seen from heading ``h`` is ``rotate(v, h)``, which maps
  ``MOVES[d]`` to ``MOVES[(d - h) % 4]`` (``test_se2.py`` checks this).
- the odometry symbol from pose t to t+1 is ``4 * turn + move`` with
  ``move = (d_t - h_t) % 4`` (forward, right, back, left in the body
  frame) and ``turn = (h_{t+1} - h_t) % 4`` (none, right, around, left).
- a query asked at step t is the relative pose of the target window seen
  from pose t: ``rotate(q - p_t, h_t)`` and ``(q_h - h_t) % 4``, and the
  target is the window rotated to face ``q_h``. By default every query is
  asked at the last step.

Recovering a pose from the odometry is a group composition: each move is
rotated by the running heading before it is summed, so a prefix sum of
the moves is not enough. ``compose`` does it in the open, for tests and
diagnostics only; a model must learn it.
"""
import numpy as np

from mapping.data import MOVES, apply_dihedral

# one quarter turn of the frame: (y, x) -> (-x, y); ROT[k] is k of them
_R1 = np.array([[0, -1], [1, 0]], dtype=np.int64)
ROT = np.stack([np.linalg.matrix_power(_R1, k) for k in range(4)])


def rotate(v, k):
    """World vectors ``v (..., 2)`` as seen from heading ``k (...)``."""
    return np.einsum('...ij,...j->...i', ROT[np.asarray(k)], v)


def unrotate(v, k):
    """Inverse of ``rotate``: body vectors back to the world frame."""
    return rotate(v, (-np.asarray(k)) % 4)


def encode(directions, headings):
    """Odometry symbols ``(..., T)`` in 0..15 from world directions
    ``(..., T)`` and headings ``(..., T + 1)``."""
    move = (directions - headings[..., :-1]) % 4
    turn = (headings[..., 1:] - headings[..., :-1]) % 4
    return 4 * turn + move


def decode(odometry):
    """``(move, turn)`` from symbols."""
    return odometry % 4, odometry // 4


def compose(odometry, step, h0=0):
    """Integrate egocentric odometry ``(..., T)`` into world displacements
    ``(..., T + 1, 2)`` from the first pose and headings ``(..., T + 1)``,
    given the first heading. Tests and diagnostics only."""
    move, turn = decode(odometry)
    h0 = np.asarray(h0)
    if h0.ndim == 0:
        h0 = np.broadcast_to(h0, move.shape[:-1])
    h0 = h0[..., None]
    headings = np.concatenate([h0, (h0 + np.cumsum(turn, axis=-1)) % 4],
                              axis=-1)
    world_dir = (move + headings[..., :-1]) % 4
    disp = np.zeros(move.shape[:-1] + (move.shape[-1] + 1, 2), np.int64)
    np.cumsum(MOVES[world_dir] * step, axis=-2, out=disp[..., 1:, :])
    return disp, headings


def face(windows, headings):
    """Windows ``(..., S, S)`` rotated to face their headings ``(...)``."""
    return apply_dihedral(windows, np.asarray(headings, dtype=np.int8))


def query_pose(queries, query_headings, anchor_positions, anchor_headings):
    """Relative query poses ``(..., Q, 3)``: body-frame (dy, dx) from the
    anchor window and the heading change, as ints. The anchors are the
    poses at each query's step, ``(..., Q, 2)`` and ``(..., Q)``."""
    body = rotate(queries - anchor_positions, anchor_headings)
    turn = (query_headings - anchor_headings) % 4
    return np.concatenate([body, turn[..., None]], axis=-1)
