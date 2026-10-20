"""Frozen hybrid7: matched 9px motion with 9x9 versus 15x15 views."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from mapping.checkpoint import load_checkpoint
from mapping.tasks import config_and_spec
from mapping.loader import load_banks
from mapping.data import make_batch,render_map,crop_windows
from mapping import se2d

out=Path('results/hybrid7_view15_controls');out.mkdir(exist_ok=False)
ck=Path('results/se2d/hybrid7/s/hybrid7_causal_inert_saw_s_agaga_100k_ieee_seed0/final.pt')
torch.set_num_threads(4);model,_,spec,_=load_checkpoint(ck,'cuda')
cfg,_=config_and_spec('se2d',32,8,1808,'uniform',9,0.,'mnist','saw',spec.offset_max);_,bank=load_banks('data/mnist.npz',cfg)
rng=np.random.default_rng(1808);halo=np.random.default_rng(1809);records={};examples={}
@torch.no_grad()
def pred(obs,odo,pose,qt,size):
 with torch.autocast('cuda',dtype=torch.bfloat16):return model(obs,odo,pose,qt,query_shape=(size,size)).float().cpu()
with torch.no_grad():
 for batchidx in range(32):
  if batchidx % 8 == 0:print(f'batch {batchidx}/32',flush=True)
  b=make_batch(rng,cfg,bank,8);largeobs=[];largetarget=[]
  for i,e in enumerate(b['episodes']):
   h,w=e.tile_grid.shape;ids=halo.integers(0,bank.shape[1],size=(h+2,w+2));sym=halo.integers(0,8,size=(h+2,w+2),dtype=np.int8);ids[1:-1,1:-1]=e.tile_grid;sym[1:-1,1:-1]=e.tile_sym;im=render_map(bank,ids,28,sym)
   obs=se2d.face(crop_windows(im,e.positions+25,15),b['headings'][i]);qh=(b['query_pose'][i,:,2]+b['headings'][i,e.query_times])%4
   tar=se2d.face(crop_windows(im,e.queries+25,15),qh);largeobs.append(obs);largetarget.append(tar)
  largeobs=np.stack(largeobs);largetarget=np.stack(largetarget)
  assert np.array_equal(largeobs[...,3:-3,3:-3],b['observations']);assert np.array_equal(largetarget[...,3:-3,3:-3],b['target'])
  # Full 15x15 targets are covered: dilating a covered 9x9 target and
  # the union of its covering 9x9 observations by 3px preserves inclusion.
  odo=torch.tensor(b['odometry'],device='cuda');obs9=torch.tensor(b['observations'],device='cuda').float()/255;obs15=torch.tensor(largeobs,device='cuda').float()/255
  src=[8,16,24,31]
  for family in ('uniform','aligned_final'):
   if family=='uniform':
    pose=b['query_pose'];qt=b['query_time'];t9=torch.tensor(b['target']).float()/255;t15=torch.tensor(largetarget).float()/255
   else:
    pose=se2d.query_pose(b['positions'][:,src],b['headings'][:,src],b['positions'][:,-1,None],b['headings'][:,-1,None]);qt=np.full((8,4),32);t9=obs9[:,src].cpu();t15=obs15[:,src].cpu()
   pose=torch.tensor(pose,device='cuda');qt=torch.tensor(qt,device='cuda')
   for name,obs,size in [('9to9',obs9,9),('15to15',obs15,15),('15to9',obs15,9),('9to15',obs9,15)]:
    for shuffled in (False,True):
     label=family+'/'+name+('/shuffled' if shuffled else '');y=pred(obs,odo.roll(1,0) if shuffled else odo,pose,qt,size);target=t15 if size==15 else t9;err=(y-target).square();center=err[...,3:-3,3:-3] if size==15 else err
     if name=='9to15':err=center  # outer target may be unseen from 9x9 history; never score it
     row={'mse':err.mean((-1,-2)).tolist(),'center_mse':center.mean((-1,-2)).tolist(),'query_time':qt.cpu().tolist()}
     if size==15 and name!='9to15':row['border_mse']=((err.sum((-1,-2))-center.sum((-1,-2)))/144).tolist()
     records.setdefault(label,[]).append(row)
     if batchidx==0 and not shuffled:examples[label]=(target.clone(),y.clone())
 summary={}
 for key,rows in records.items():
  d={}
  for metric in ('mse','center_mse','border_mse'):
   if metric not in rows[0]:continue
   a=np.concatenate([r[metric] for r in rows]);d[metric]=float(a.mean());d[metric+'_episode_se']=float(a.mean(1).std(ddof=1)/16)
   qt=np.concatenate([r['query_time'] for r in rows]);d[metric+'_last']=float(a[qt==32].mean()) if (qt==32).any() else None
  summary[key]=d
 result={'checkpoint':str(ck),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'seed':1808,'halo_seed':1809,'episodes':256,'steps':32,'step_px':9,'notes':['9to15 scores only the supported central9x9; no full-view or border metric for that condition.'],'summary':summary,'batches':records}
 (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2),flush=True)
 # Same preselected queries; no quality-based selection.
 canvas=Image.new('RGB',(780,1030),'#222222');draw=ImageDraw.Draw(canvas)
 draw.text((10,10),'Frozen hybrid7 S | same 9px trajectories, textures and query centers',fill='white')
 draw.text((10,28),'Target 15 | Native 9 | 15 input/output | 15 input, 9 output',fill='white')
 for r in range(8):
  family='uniform' if r<4 else 'aligned_final';q=r%4;i=0;y0=65+r*119
  draw.text((5,y0),f'{family}\nquery {q}',fill='white')
  target15=examples[family+'/15to15'][0][i,q]
  arrays=[target15,examples[family+'/9to9'][1][i,q],examples[family+'/15to15'][1][i,q],examples[family+'/15to9'][1][i,q]]
  for col,a in enumerate(arrays):
   # Identical scale: every pixel is a 6x6 square. 9x9 views centered.
   n=a.shape[-1];im=Image.fromarray((a.clamp(0,1).numpy()*255).astype('uint8')).resize((n*6,n*6),Image.Resampling.NEAREST);x=150+col*155+(90-n*6)//2;yy=y0+(90-n*6)//2;canvas.paste(im,(x,yy))
   if col:
    target=target15 if n==15 else target15[3:-3,3:-3];draw.text((145+col*155,y0+94),f'MSE {float((a-target).square().mean()):.4f}',fill='white')
 canvas.save(out/'renders.png')
