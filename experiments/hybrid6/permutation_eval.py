"""Frozen history patch-order probe; pixels and camera metadata move together."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from mapping.checkpoint import load_checkpoint
from mapping.tasks import config_and_spec
from mapping.loader import load_banks
from mapping.data import make_batch
from mapping import se2d
from mapping.hybrid6_geometry import patchify,unpatchify,patch_calibration

out=Path('results/hybrid6_permutation');out.mkdir(exist_ok=False)
ck=Path('results/se2d/hybrid6_causal/s/local_causal_inert_saw_s_100k_ieee_seed0/final.pt')
torch.set_num_threads(4);model,_,spec,_=load_checkpoint(ck,'cuda');cfg,_=config_and_spec('se2d',32,8,1810,'uniform',9,0.,'mnist','saw',spec.offset_max);_,bank=load_banks('data/mnist.npz',cfg)
rng=np.random.default_rng(1810);prng=np.random.default_rng(1811);fixed=prng.permutation(9);records={};pictures={};identity_max=0.;restore_max=0.
@torch.no_grad()
def camera(obs,odo,qp,qt,perm):
 B,F=obs.shape[:2];Q=qp.shape[1];pixels=patchify(obs);poses=torch.eye(4,device='cuda').expand(B,F,9,4,4);cal=patch_calibration(9,9,'cuda').expand(B,F,9,5)
 transitions,query,known=model._task_poses(odo,qp)
 if perm is not None:
  ix=torch.tensor(perm,device='cuda').long().expand(B,F,9)
  pixels=pixels.gather(2,ix[...,None].expand(-1,-1,-1,9));cal=cal.gather(2,ix[...,None].expand(-1,-1,-1,5));poses=poses.gather(2,ix[...,None,None].expand(-1,-1,-1,4,4))
  inv=ix.argsort(-1);assert torch.equal(pixels.gather(2,inv[...,None].expand(-1,-1,-1,9)),patchify(obs));assert torch.equal(cal.gather(2,inv[...,None].expand(-1,-1,-1,5)),patch_calibration(9,9,'cuda').expand(B,F,9,5))
 with torch.autocast('cuda',dtype=torch.bfloat16):
  y=model.forward_cameras(pixels,poses,cal,transitions,query[:,:,None].expand(B,Q,9,4,4),patch_calibration(9,9,'cuda').expand(B,Q,9,5),qt,transition_known=known)
 return unpatchify(y,9,9).float().cpu()
with torch.no_grad():
 for bi in range(32):
  b=make_batch(rng,cfg,bank,8);obs=torch.tensor(b['observations'],device='cuda').float()/255;odo=torch.tensor(b['odometry'],device='cuda');src=[8,16,24,31]
  identity=np.broadcast_to(np.arange(9),(8,33,9)).copy();last=identity.copy();last[:,-1]=np.arange(8,-1,-1);adj=np.arange(9);adj[:2]=[1,0]
  perms={'identity':None,'adjacent_swap':adj,'column_major':np.arange(9).reshape(3,3).T.flatten(),'reverse':np.arange(8,-1,-1).copy(),'fixed_random':fixed,'independent_frame_random':np.stack([prng.permutation(9) for _ in range(8*33)]).reshape(8,33,9),'reverse_last_frame':last}
  for family in ('uniform','aligned_final'):
   if family=='uniform':qp=b['query_pose'];qt=b['query_time'];target=torch.tensor(b['target']).float()/255
   else:qp=se2d.query_pose(b['positions'][:,src],b['headings'][:,src],b['positions'][:,-1,None],b['headings'][:,-1,None]);qt=np.full((8,4),32);target=obs[:,src].cpu()
   qp=torch.tensor(qp,device='cuda');qt=torch.tensor(qt,device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):native=model(obs,odo,qp,qt).float().cpu()
   for name,perm in perms.items():
    y=camera(obs,odo,qp,qt,perm)
    if name=='identity':identity_max=max(identity_max,float((y-native).abs().max()));assert torch.equal(y,native)
    row={'mse':(y-target).square().mean((-1,-2)).tolist(),'change_mse':(y-native).square().mean((-1,-2)).tolist(),'query_time':qt.cpu().tolist()};records.setdefault(family+'/'+name,[]).append(row)
    if bi==0 and family=='aligned_final':pictures[name]=(target.clone(),y.clone())
   y=camera(obs,odo.roll(1,0),qp,qt,None);records.setdefault(family+'/shuffled_odometry',[]).append({'mse':(y-target).square().mean((-1,-2)).tolist(),'change_mse':(y-native).square().mean((-1,-2)).tolist(),'query_time':qt.cpu().tolist()})
summary={}
for key,rows in records.items():
 e=np.concatenate([r['mse'] for r in rows]);delta=np.concatenate([r['change_mse'] for r in rows]);times=np.concatenate([r['query_time'] for r in rows]);summary[key]={'mse':float(e.mean()),'episode_se':float(e.mean(1).std(ddof=1)/16),'prediction_change_mse':float(delta.mean()),'last_time_mse':float(e[times==32].mean())}
result={'checkpoint':str(ck),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'seed':1810,'permutation_seed':1811,'fixed_permutation':fixed.tolist(),'episodes':256,'identity_native_max_abs':identity_max,'summary':summary,'batches':records};(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
labels=['Target','identity','adjacent_swap','column_major','reverse','fixed_random','independent_frame_random','reverse_last_frame'];canvas=Image.new('RGB',(1160,610),'#222222');d=ImageDraw.Draw(canvas);d.text((10,8),'Same nine patches and calibration; only history serialization changes | first two episodes',fill='white')
for col,label in enumerate(labels):d.text((90+col*132,32),label.replace('_','\n'),fill='white')
for r in range(8):
 episode=r//4;q=r%4;y0=82+r*64;d.text((5,y0),f'ep{episode} s{src[q]}',fill='white')
 for col,label in enumerate(labels):
  a=pictures['identity'][0][episode,q] if col==0 else pictures[label][1][episode,q];im=Image.fromarray((a.clamp(0,1).numpy()*255).astype('uint8')).resize((54,54),Image.Resampling.NEAREST);canvas.paste(im,(105+col*132,y0))
canvas.save(out/'renders.png')
