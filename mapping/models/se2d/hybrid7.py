"""Hybrid7: block-causal attention followed by register-write-only GDN.

Per timestep: five ordered registers, an unordered observation block, outgoing
odometry. Queries read through observations, never outgoing odometry. Only
registers enter the short convolution or update/decay recurrent state; all other
tokens read it passively. GDN convolution carries across timestep boundaries.
Camera features and task adapters are inherited unchanged from hybrid6.
"""
import sys
import torch
from torch import nn
from torch.nn import functional as F
from mapping.models.se2d import register
from mapping.models.se2d.hybrid6 import Hybrid6, LR, SIZES
from mapping.models.se2d.hybrid4 import (BASE_WIDTH, HEAD_DIM, Attention, Block,
                                       rope_cos_sin, GatedDeltaNet)
from mapping.models.se2d.hybrid4_causal import CausalGatedDeltaNet
from mapping.models.query_mode import QueryLayout, block_mask
from mapping.models.scan_precision import set_scan_precision
from mapping.hybrid6_geometry import pose_features, sensor_features

SCALING_RULE = ('Hybrid6 width ladder with attention between GDN layers and final attention, '
                'SwiGLU, five register writes per timestep, and block-causal '
                'attention; six GDN heads of dimension32 independent of width; '
                'patch counts change no parameter shapes.')
ARCH = {'gdn_heads': 6, 'final_attention': True, 'inter_gdn_attention': True}


class RegisterLayout(QueryLayout):
    """Separate attention-history membership from recurrent write membership."""
    def __init__(self, order, n_hist, q, per_query, phases, registers):
        super().__init__(order, n_hist, q, per_query)
        self.phases = phases.gather(1, order)
        self.hist = registers.gather(1, order)  # GDN write/conv tokens only
        # Same number of registers in each example, in temporal/slot order.
        self.hist_idx = order.argsort(1).gather(1, registers[0].nonzero().flatten()[None].expand(order.shape[0], -1))

    def mask_mod(self, causal=True):
        qid, phase = self.qid, self.phases
        def mod(b, h, qi, ki):
            return ((qid[b, ki] == 0) & (phase[b, ki] <= phase[b, qi])) | ((qid[b, qi] > 0) & (qid[b, qi] == qid[b, ki]))
        return mod

    def block_mask(self, causal=True):
        if 'phase' not in self._masks:
            b, length = self.qid.shape
            self._masks['phase'] = block_mask(self.mask_mod(), b, length, self.qid.device)
        return self._masks['phase']


class Hybrid7(Hybrid6):
    QUERY_MODES = ('inert',)
    SCAN_PRECISIONS = ('tf32', 'ieee')

    def __init__(self, spec, width, query_mode='mixed', scan_precision='tf32', registers=5, gdn_heads=6, final_attention=True, inter_gdn_attention=False, odometry_writes=False):
        if registers != 5:
            raise ValueError('hybrid7 initial design fixes five registers per timestep')
        super().__init__(spec, width, query_mode=query_mode, causal=True,
                         scan_precision=scan_precision)
        self.odometry_writes = odometry_writes
        self.register_count = registers
        self.registers = nn.Embedding(registers, width)
        nn.init.normal_(self.registers.weight, std=.02)
        # Attention must precede recurrence; swap the final pair for retrieval.
        heads = mixers = width // HEAD_DIM
        if final_attention and mixers < 4:
            raise ValueError('final-attention Hybrid7 requires size S or larger')
        kinds = [Attention if i % 2 == 0 else CausalGatedDeltaNet
                 for i in range(mixers)]
        if final_attention:
            kinds[-2], kinds[-1] = kinds[-1], kinds[-2]
        if inter_gdn_attention:
            if not final_attention:
                raise ValueError('inter-GDN attention requires final_attention')
            kinds.insert(-2, Attention)
        mult = BASE_WIDTH / width
        std = .5 / width ** .5
        out_std = std / (2 * mixers) ** .5
        self.blocks = nn.ModuleList([
            Block(kind(
                width, heads if kind is Attention else gdn_heads, mult, std, out_std),
                width, mult, std, out_std)
            for kind in kinds])
        set_scan_precision(self, scan_precision, GatedDeltaNet)

    def forward_cameras(self, pixels, obs_pose, obs_calibration, transitions,
                        query_pose, query_calibration, query_time=None, *,
                        transition_known=None):
        if pixels.ndim != 4 or pixels.shape[-1] != 9:
            raise ValueError('pixels must be (B, frames, patches, 9)')
        b, frames, po, _ = pixels.shape
        if frames < 1 or po < 1:
            raise ValueError('at least one frame and observation patch required')
        if query_pose.ndim != 5 or query_pose.shape[0] != b:
            raise ValueError('query_pose must be (B, queries, patches, 4, 4)')
        q, pq = query_pose.shape[1:3]
        if pq < 1:
            raise ValueError('at least one query patch required')
        expected = ((obs_pose, (b, frames, po, 4, 4)),
                    (obs_calibration, (b, frames, po, 5)),
                    (transitions, (b, frames-1, 4, 4)),
                    (query_pose, (b, q, pq, 4, 4)),
                    (query_calibration, (b, q, pq, 5)))
        for tensor, shape in expected:
            if tensor.shape != shape:
                raise ValueError(f'expected shape {shape}, got {tuple(tensor.shape)}')
        if transition_known is not None and transition_known.shape != (b, frames-1):
            raise ValueError('transition_known must be (B, frames-1)')
        if query_time is None:
            query_time = torch.full((b, q), frames-1, device=pixels.device, dtype=torch.long)
        if query_time.shape != (b, q) or query_time.dtype not in (torch.int32, torch.int64):
            raise ValueError('query_time must be integer (B, queries)')
        if ((query_time < 0) | (query_time >= frames)).any():
            raise ValueError('query_time outside observation sequence')

        def pose_embedding(p, known=None):
            return self.pose_embed(pose_features(p, known).to(self.pose_embed.weight.dtype))
        def sensor_embedding(c):
            return self.sensor_embed(sensor_features(c).to(self.sensor_embed.weight.dtype))
        obs = (self.patch_embed(pixels.to(self.patch_embed.weight.dtype))
               + pose_embedding(obs_pose) + sensor_embedding(obs_calibration)
               + self.roles.weight[0])
        odo = pose_embedding(transitions, transition_known) + self.roles.weight[2]
        qry = pose_embedding(query_pose) + sensor_embedding(query_calibration) + self.roles.weight[1]
        k = self.register_count
        reg = self.registers.weight[None, None].expand(b, frames, k, -1)
        seq = torch.cat((reg, obs, F.pad(odo, (0,0,0,1))[:, :, None]), 2).flatten(1,2)[:, :-1]
        n_hist = seq.shape[1]
        x = torch.cat((seq, qry.flatten(1,2)), 1)
        steps = torch.arange(frames, device=x.device)
        offsets = torch.cat((torch.zeros(k, device=x.device, dtype=torch.long),
                             torch.ones(po, device=x.device, dtype=torch.long),
                             torch.full((1,), 2, device=x.device, dtype=torch.long)))
        hp = (3 * steps[:, None] + offsets).flatten()[:-1]
        qp = (3 * query_time[..., None] + 1).expand(b,q,pq).flatten(1)
        phases = torch.cat((hp[None].expand(b,-1), qp), 1)
        # Stable within each phase: observation slots before same-phase queries.
        order = (phases * x.shape[1] + torch.arange(x.shape[1], device=x.device)).argsort(1)
        write_offsets = (offsets == 0) | ((offsets == 2) & self.odometry_writes)
        writes = torch.cat((write_offsets[None].expand(frames,-1).flatten()[:-1][None].expand(b,-1),
                            torch.zeros(b,q*pq,device=x.device,dtype=torch.bool)),1)
        layout = RegisterLayout(order, n_hist, q, pq, phases, writes)
        x = x.gather(1, order[...,None].expand(-1,-1,self.width))
        cos, sin = rope_cos_sin(layout.phases // 3, HEAD_DIM)
        for block in self.blocks:
            x = block(x, cos, sin, layout)
        x = x.gather(1, order.argsort(1)[...,None].expand(-1,-1,self.width))
        return self.readout(self.norm(x[:,n_hist:])).reshape(b,q,pq,9)


def build(size, spec):
    return Hybrid7(spec, **SIZES[size], **ARCH, query_mode='inert')

register('hybrid7')(sys.modules[__name__])
