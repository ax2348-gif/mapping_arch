"""Checkpoint-independent frozen-history replay for four-block inert Hybrid4/Hybrid6 (9x9 SE2D views)."""
import torch
from torch.nn import functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from mapping.models.se2d.hybrid4 import apply_rope, rope_cos_sin
from mapping.models.scan_precision import triton_f32
from mapping.hybrid6_geometry import planar_pose,pose_features,sensor_features,patch_calibration

@torch.no_grad()
def native(model,b):
 with torch.autocast('cuda',dtype=torch.bfloat16):return model(b['observations'].cuda().float()/255,b['odometry'].cuda(),b['query_pose'].cuda(),b['query_time'].cuda()).float().cpu()

@torch.no_grad()
def query_input(model,pose):
 pose=pose.cuda();B,Q=pose.shape[:2]
 if not hasattr(model,'pose_embed'):
  with torch.autocast('cuda',dtype=torch.bfloat16):return (model.offset_embed(model.offset_features(pose[...,:2]))[:,:,None]+model.turn_emb(pose[...,2])[:,:,None]+model.query_pos(torch.arange(9,device='cuda'))).flatten(1,2)
 physical=torch.cat([pose[...,:2].float(),pose[...,2,None].float()*(torch.pi/2)],-1)
 pp=planar_pose(physical)[:,:,None].expand(B,Q,9,4,4)
 cal=patch_calibration(9,9,'cuda').expand(B,Q,9,5)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  return (model.pose_embed(pose_features(pp).to(model.pose_embed.weight.dtype))+model.sensor_embed(sensor_features(cal).to(model.sensor_embed.weight.dtype))+model.roles.weight[1]).flatten(1,2)

@torch.no_grad()
def capture(model,b):
 T=b['observations'].shape[1]-1;Q=b['query_pose'].shape[1]
 assert torch.all(b['query_time']==T), 'Replay requires all queries after final observation'
 mem={'states':{},'kv':{},'time':T};hooks=[];restore=[]
 def first(mod,args):mem['input']=args[0][:,-9*Q:].detach().clone()
 hooks.append(model.blocks[0].register_forward_pre_hook(first))
 for j in [0,2]:
  mix=model.blocks[j].mixer;restore.append((mix,mix.scan))
  def scan(q,k,v,g,beta,j=j):
   if getattr(model,'scan_precision','tf32')=='ieee':
    with torch.autocast('cuda',enabled=False),triton_f32('ieee'):
     out,state=chunk_gated_delta_rule(*[a.float().contiguous() for a in (q,k,v,g,beta)],scale=1.,chunk_size=32,output_final_state=True)
   else:out,state=chunk_gated_delta_rule(*[a.contiguous() for a in (q,k,v,g,beta)],scale=1.,output_final_state=True)
   mem['states'][j]=state.detach();return out.to(v.dtype)
  mix.scan=scan
 for j in [1,3]:
  def pre(mod,args,j=j):
   x,cos,sin,layout=args;m=layout.gather(x);B,L,W=m.shape;_,k,v=mod.qkv(m).view(B,L,3,4,32).unbind(2);c,s=rope_cos_sin((torch.arange(L,device='cuda')//10)[None].expand(B,-1),32);mem['kv'][j]=(apply_rope(k,c,s).detach(),v.detach())
   if j==3:
    mem['raw_k']=k.detach();q=mod.qkv(x).view(B,x.shape[1],3,4,32)[:,:,0];mem['native_q']=apply_rope(q,cos,sin)[:,-9*Q:].reshape(B,Q,9,4,32).float().cpu()
  hooks.append(model.blocks[j].mixer.register_forward_pre_hook(pre))
 try:pred=native(model,b)
 finally:
  for h in hooks:h.remove()
  for mix,scan in restore:mix.scan=scan
 assert torch.equal(query_input(model,b['query_pose']),mem['input'])
 return mem,pred

@torch.no_grad()
def replay(model,mem,final_only=False,zero_states=(),zero_head=None,skip_mlp=None):
 B=mem['input'].shape[0];Q=mem['input'].shape[1]//9;x=mem['input'].reshape(B*Q,9,128).clone();qfinal=None;maps=None
 def expand(t):return t[:,None].expand(-1,Q,*t.shape[1:]).reshape(B*Q,*t.shape[1:])
 c,s=rope_cos_sin(torch.full((B*Q,9),mem['time'],device='cuda'),32)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for j,block in enumerate(model.blocks):
   z=block.norm1(x);mix=block.mixer
   if j%2==0:
    raw=F.silu(mix.q(z)).view(B*Q,9,4,32);q=F.normalize(raw.float(),dim=-1).to(raw.dtype)*32**-.5;state=mem['states'][j];state=torch.zeros_like(state) if j in zero_states else state
    # The native scan accumulates against FP32 state before casting its output.
    with torch.autocast('cuda',enabled=False):o=torch.einsum('bnhk,bhkv->bnhv',q.float(),expand(state).float()).to(q.dtype)
    o=mix.norm(o).to(z.dtype)*F.silu(mix.gate(z)).view(B*Q,9,4,32)
   else:
    q,k,v=mix.qkv(z).view(B*Q,9,3,4,32).unbind(2);q=apply_rope(q,c,s);k=apply_rope(k,c,s);mk,mv=map(expand,mem['kv'][j]);L=mk.shape[1];kk=torch.cat([mk,k],1);vv=torch.cat([mv,v],1);mask=torch.ones(9,L+9,device='cuda',dtype=torch.bool);mask[:,L:]=torch.ones(9,9,device='cuda',dtype=torch.bool).tril()
    if final_only and j==1:mask[:,:L]=False
    if j==3:
     qfinal=q.reshape(B,Q,9,4,32).float().cpu()
     with torch.autocast('cuda',enabled=False):maps=(torch.einsum('bnhd,bmhd->bhnm',q.float(),kk.float())/32**.5).masked_fill(~mask[None,None],float('-inf')).softmax(-1).reshape(B,Q,4,9,-1).cpu()
    o=F.scaled_dot_product_attention(q.transpose(1,2),kk.transpose(1,2),vv.transpose(1,2),attn_mask=mask).transpose(1,2)
   if zero_head is not None and zero_head[0]==j:o=o.clone();o[:,:,zero_head[1]]=0
   x=x+mix.out(o.reshape(B*Q,9,128))
   if j!=skip_mlp:x=x+block.mlp(block.norm2(x))
  y=model.readout(model.norm(x)).view(B,Q,3,3,3,3).permute(0,1,2,4,3,5).reshape(B,Q,9,9)
 return y.float().cpu(),maps,qfinal


def metrics(pred,target,maps=None,sources=None):
 out={'pixel_mse':float((pred-target).square().mean())}
 if maps is not None:
  expected=torch.tensor(sources)[None,:,None]*10+torch.arange(9)[None,None];top=maps.argmax(-1);out['patch_accuracy_by_head']=[float((top[:,:,h]==expected).float().mean()) for h in range(4)]
 return out
