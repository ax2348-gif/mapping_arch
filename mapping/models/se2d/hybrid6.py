"""hybrid6: hybrid4 mixers with calibrated camera patches and three roles.

The task adapter accepts existing SE2d inputs. forward_cameras is the explicit
local SE3 camera interface (fixed 3x3 grayscale samples, dense token batches).
Observation/render share pose and calibration encoders; transitions share the
pose encoder. No learned patch identities, no world poses, no pose integration.

Audit:
1. A discrete odometry symbol is decoded into its ONE local move/turn; symbols
   are never accumulated. Unknown symbol 16 has masked pose features and a
   known/unknown scalar, not a fabricated identity measurement.
2. Raw query poses become local matrices. Image extrinsics are shared across
   patches. Crop principal points depend only on the image grid, never motion.
3. Fourier features embed raw translations/principal points; rotation matrices
   and log focal lengths are calibration features, not geometric attention.
4. Time enters sequence placement and temporal RoPE only. Hybrid4's unchanged
   causal/bidirectional mixers and inert layout control access to history.
5. forward_cameras consumes only explicitly supplied local camera poses,
   intrinsics, pixels, and local transitions; it never renders rays or consumes
   scene depths. Additive embeddings combine attributes, not transformations.
"""
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from mapping.hybrid6_geometry import (POSE_FEATURES, SENSOR_FEATURES,
    patch_calibration, patchify, planar_pose, pose_features, sensor_features,
    unpatchify)
from mapping.models.query_mode import QueryLayout, check_query_mode
from mapping.models.scan_precision import check_scan_precision, set_scan_precision
from mapping.models.se2d import register
from mapping.models.se2d.hybrid4 import (BASE_WIDTH, HEAD_DIM, LR, SIZES,
    Attention, Block, GatedDeltaNet, MuLinear, rope_cos_sin)
from mapping.models.se2d.hybrid4_causal import CausalAttention, CausalGatedDeltaNet

SCALING_RULE = ('Hybrid4 width ladder: heads and mixer depth = width/32, '
                'alternating delta-rule and attention blocks, MLP ratio 4; '
                'same muP width multipliers and depth initialization. Fixed '
                '3x3 patches and fixed-dimensional SE3/calibration features; '
                'three role vectors, no resolution-dependent parameters.')
ARCH = {}


def token_order(query_time, frames, obs_patches, query_patches):
    b, q = query_time.shape
    steps = torch.arange(frames, device=query_time.device)
    context = torch.cat(((steps[:, None] * 3).expand(-1, obs_patches),
                         steps[:, None] * 3 + 2), 1).flatten()[:-1]
    queries = (query_time[..., None] * 3 + 1).expand(b, q, query_patches).flatten(1)
    key = torch.cat((context[None].expand(b, -1), queries), 1)
    n = key.shape[1]
    return (key * n + torch.arange(n, device=key.device)).argsort(1)


class Hybrid6(nn.Module):
    QUERY_MODES = ('mixed', 'inert')
    SCAN_PRECISIONS = ('tf32', 'ieee')
    continuous = False

    def __init__(self, spec, width, query_mode='mixed', causal=False, scan_precision='tf32'):
        super().__init__()
        check_query_mode(self, query_mode)
        check_scan_precision(self, scan_precision)
        if width < 64 or width % (2 * HEAD_DIM):
            raise ValueError('width must be a positive multiple of 64')
        self.spec, self.width = spec, width
        self.query_mode, self.causal = query_mode, causal
        self.scan_precision = scan_precision
        self.patch_embed = nn.Linear(9, width)
        self.pose_embed = nn.Linear(POSE_FEATURES, width)
        self.sensor_embed = nn.Linear(SENSOR_FEATURES, width)
        self.roles = nn.Embedding(3, width)  # observation, render, transition
        nn.init.normal_(self.roles.weight, std=0.02)
        heads = mixers = width // HEAD_DIM
        mult = BASE_WIDTH / width
        std = 0.5 / math.sqrt(width)
        out_std = std / math.sqrt(2 * mixers)
        delta = CausalGatedDeltaNet if causal else GatedDeltaNet
        attention = CausalAttention if causal else Attention
        self.blocks = nn.ModuleList([
            Block((delta if i % 2 == 0 else attention)(
                width, heads, mult, std, out_std), width, mult, std, out_std)
            for i in range(mixers)])
        self.norm = nn.LayerNorm(width)
        self.readout = MuLinear(width, 9, mult, std, zero=True)
        set_scan_precision(self, scan_precision, GatedDeltaNet)

    def _task_poses(self, odometry, query_pose):
        if self.continuous:
            return planar_pose(odometry), planar_pose(query_pose), None
        if odometry.dtype not in (torch.int32, torch.int64):
            raise ValueError('SE2d odometry must be integer symbols')
        if ((odometry < 0) | (odometry >= self.spec.odometry)).any():
            raise ValueError('odometry symbol out of range')
        turn = query_pose[..., 2]
        if ((turn < 0) | (turn > 3) | (turn != turn.round())).any():
            raise ValueError('SE2d query turns must be integers in 0..3')
        known = odometry != 16
        symbol = torch.where(known, odometry, 0)
        # Literal local move vocabulary, not integrated trajectory positions.
        moves = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]],
                             device=odometry.device, dtype=torch.float32)
        body = moves[symbol % 4] * self.spec.step_px
        odo = torch.cat((body, (symbol // 4)[..., None] * (math.pi / 2)), -1)
        qry = torch.cat((query_pose[..., :2].float(), turn[..., None] * (math.pi / 2)), -1)
        return planar_pose(odo), planar_pose(qry), known

    def forward(self, obs, odometry, query_pose, query_time=None, *, query_shape=None):
        """Existing task interface; optional independent (height,width) output grid.

        Both grids are runtime-sized; the standard harness still supplies its
        original fixed dimensions and f=1. Scale/multiview use forward_cameras.
        """
        if obs.ndim != 4:
            raise ValueError('obs must be (B, frames, height, width)')
        b, frames, h, w = obs.shape
        if query_pose.ndim != 3 or query_pose.shape[0] != b or query_pose.shape[-1] != 3:
            raise ValueError('query_pose must be (B, queries, 3)')
        if odometry.shape[:2] != (b, frames - 1):
            raise ValueError('odometry must have one transition between frames')
        q = query_pose.shape[1]
        oh, ow = (h, w) if query_shape is None else query_shape
        obs_cal = patch_calibration(h, w, obs.device)
        qry_cal = patch_calibration(oh, ow, obs.device)
        po, pq = len(obs_cal), len(qry_cal)
        odo, qry, known = self._task_poses(odometry, query_pose)
        identity = torch.eye(4, device=obs.device)
        patches = self.forward_cameras(
            patchify(obs), identity.expand(b, frames, po, 4, 4),
            obs_cal.expand(b, frames, po, 5), odo,
            qry[:, :, None].expand(b, q, pq, 4, 4),
            qry_cal.expand(b, q, pq, 5), query_time, transition_known=known)
        return unpatchify(patches, oh, ow)

    def forward_cameras(self, pixels, obs_pose, obs_calibration, transitions,
                        query_pose, query_calibration, query_time=None, *,
                        transition_known=None):
        """Local camera interface, fixed 3x3 grayscale patches.

        pixels: (B,F,Pobs,9); obs_pose: (B,F,Pobs,4,4);
        obs_calibration: (B,F,Pobs,5); transitions: (B,F-1,4,4);
        query_pose: (B,Q,Pquery,4,4); query_calibration: (B,Q,Pquery,5).
        Returns (B,Q,Pquery,9). Multiview cameras can share F and occupy patch
        slots with their own poses/calibration. No ragged/padded slots: every
        supplied observation slot is a real measurement and memory update.
        Caller supplies valid rigid matrices in the documented convention.
        """
        if pixels.ndim != 4 or pixels.shape[-1] != 9:
            raise ValueError('pixels must be (B, frames, patches, 9)')
        b, frames, po, _ = pixels.shape
        if frames < 1 or po < 1:
            raise ValueError('at least one frame and observation patch required')
        if query_pose.ndim != 5 or query_pose.shape[0] != b:
            raise ValueError('query_pose must be (B, queries, patches, 4, 4)')
        q, pq = query_pose.shape[1:3]
        if pq < 1:
            raise ValueError('at least one patch per query required')
        expected = ((obs_pose, (b, frames, po, 4, 4)),
                    (obs_calibration, (b, frames, po, 5)),
                    (transitions, (b, frames - 1, 4, 4)),
                    (query_pose, (b, q, pq, 4, 4)),
                    (query_calibration, (b, q, pq, 5)))
        for tensor, shape in expected:
            if tensor.shape != shape:
                raise ValueError(f'expected shape {shape}, got {tuple(tensor.shape)}')
        if transition_known is not None and transition_known.shape != (b, frames - 1):
            raise ValueError('transition_known must be (B, frames-1)')
        if query_time is None:
            query_time = torch.full((b, q), frames - 1, device=pixels.device, dtype=torch.long)
        if query_time.shape != (b, q) or query_time.dtype not in (torch.int32, torch.int64):
            raise ValueError('query_time must be integer (B, queries)')
        if ((query_time < 0) | (query_time >= frames)).any():
            raise ValueError('query_time outside observation sequence')
        # Feature extraction stays fp32; cast only at learned projections for
        # explicit bf16 modules as well as the harness autocast path.
        def pose_embedding(p, known=None):
            return self.pose_embed(pose_features(p, known).to(self.pose_embed.weight.dtype))
        def sensor_embedding(c):
            return self.sensor_embed(sensor_features(c).to(self.sensor_embed.weight.dtype))
        obs = (self.patch_embed(pixels.to(self.patch_embed.weight.dtype))
               + pose_embedding(obs_pose) + sensor_embedding(obs_calibration)
               + self.roles.weight[0])
        odo = pose_embedding(transitions, transition_known) + self.roles.weight[2]
        qry = pose_embedding(query_pose) + sensor_embedding(query_calibration) + self.roles.weight[1]
        seq = torch.cat((obs, F.pad(odo, (0, 0, 0, 1))[:, :, None]), 2).flatten(1, 2)[:, :-1]
        context_len = frames * (po + 1) - 1
        x = torch.cat((seq, qry.flatten(1, 2)), 1)
        steps = torch.arange(frames, device=pixels.device)
        context_pos = steps[:, None].expand(frames, po + 1).flatten()[:-1]
        pos = torch.cat((context_pos[None].expand(b, -1),
                         query_time[..., None].expand(b, q, pq).flatten(1)), 1)
        order = token_order(query_time, frames, po, pq)
        x = x.gather(1, order[..., None].expand(-1, -1, self.width))
        pos = pos.gather(1, order)
        cos, sin = rope_cos_sin(pos, HEAD_DIM)
        layout = QueryLayout(order, context_len, q, pq) if self.query_mode == 'inert' else None
        for block in self.blocks:
            x = block(x, cos, sin, layout)
        x = x.gather(1, order.argsort(1)[..., None].expand(-1, -1, self.width))
        y = self.readout(self.norm(x[:, context_len:]))
        return y.reshape(b, q, pq, 9)


def build(size, spec):
    return Hybrid6(spec, **SIZES[size], **ARCH)


register('hybrid6')(sys.modules[__name__])
