"""Small CPU diagnostic of saved checkpoints; leaves live training untouched."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from mapping.checkpoint import load_checkpoint
from mapping.loader import load_banks
from mapping.data import make_batch
from mapping.tasks import config_and_spec
import mapping.models.se2d.hybrid4 as h

torch.set_num_threads(2)
ck=sys.argv[1];model,args,spec,task=load_checkpoint(ck,'cpu')
for m in model.modules():
 if isinstance(m,h.GatedDeltaNet):m.scan=h.delta_scan
cfg,_=config_and_spec('se2d',32,8,1821,'uniform',9,0.,'mnist','saw',spec.offset_max)
_,bank=load_banks('data/mnist.npz',cfg);rng=np.random.default_rng(1821)
state={};routing={};betas={};rows=[]
def dense(q,k,v,mask):
 b,heads,n,d=q.shape
 allowed=mask.mask_mod(torch.arange(b)[:,None,None,None],torch.arange(heads)[None,:,None,None],torch.arange(n)[None,None,:,None],torch.arange(n)[None,None,None,:])
 prob=(q@k.transpose(-1,-2)/d**.5).masked_fill(~allowed,float('-inf')).softmax(-1)
 if state.get('collect'):
  j=state['layer'];layout=state['layout'];phase=layout.phases;qid=layout.qid
  regs=layout.hist & (phase>=3)
  incoming=(phase[:,None,:]==phase[:,:,None]-1)&(qid[:,None,:]==0)
  allodo=(phase%3==2)&(qid==0)
  vals={'incoming_odometry_mass':(prob*incoming[:,None]).sum(-1),'all_odometry_mass':(prob*allodo[:,None,None,:]).sum(-1)}
  routing.setdefault(j,{})
  for key,z in vals.items():routing[j].setdefault(key,[]).append(z.transpose(1,2)[regs].mean(0).tolist())
 if state.get('erase_layer')==state.get('layer'):
  layout=state['layout'];phase=layout.phases
  edge=layout.hist[:,:,None] & (phase[:,None,:]==phase[:,:,None]-1) & (layout.qid[:,None,:]==0)
  prob=prob.masked_fill(edge[:,None],0)
  prob=prob/prob.sum(-1,keepdim=True).clamp_min(1e-12)
 return prob@v
h.attend=dense
for j,block in enumerate(model.blocks):
 if isinstance(block.mixer,h.Attention):
  def pre(m,a,j=j):state.update(layer=j,layout=a[-1])
  block.mixer.register_forward_pre_hook(pre)
 else:
  def pre(m,a,j=j):
   if state.get('collect'):
    x,layout=a;beta=(2*torch.sigmoid(m.beta(x).float()))[layout.hist];betas.setdefault(j,[]).append({'mean':beta.mean(0).tolist(),'gt1_9':(beta>1.9).float().mean(0).tolist()})
  block.mixer.register_forward_pre_hook(pre)
with torch.no_grad():
 for _ in range(4):
  batch={k:torch.from_numpy(v) for k,v in make_batch(rng,cfg,bank,4).items() if isinstance(v,np.ndarray)}
  obs=batch['observations'].float()/255;odo=batch['odometry'];qp=batch['query_pose'];qt=batch['query_time']
  target=batch['target'].float()/255 if 'target'in batch else batch['targets'].float()/255
  state['collect']=True;y=model(obs,odo,qp,qt);state['collect']=False
  row={'native_mse':float((y-target).square().mean())}
  for j,block in enumerate(model.blocks):
   if isinstance(block.mixer,h.GatedDeltaNet):
    handle=block.mixer.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    z=model(obs,odo,qp,qt);handle.remove();row[f'zero_gdn{j}_mse']=float((z-target).square().mean());row[f'zero_gdn{j}_prediction_change']=float((z-y).square().mean())
  for j in (0,2):
   state['erase_layer']=j;z=model(obs,odo,qp,qt);state.pop('erase_layer')
   row[f'erase_incoming_odo_attention{j}_mse']=float((z-target).square().mean())
   row[f'erase_incoming_odo_attention{j}_prediction_change']=float((z-y).square().mean())
  rows.append(row)
out={'checkpoint':ck,'episodes':16,'seed':1821,'backend':'CPU FP32 reference scan and dense attention; descriptive screen, not GPU replay validation','metrics':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},'routing':{j:{k:np.mean(v,axis=0).tolist() for k,v in a.items()} for j,a in routing.items()},'beta':{j:{k:np.mean([a[k] for a in arr],axis=0).tolist() for k in arr[0]} for j,arr in betas.items()}}
p=Path('results/hybrid7_routing')/Path(ck).stem;p.mkdir(parents=True,exist_ok=True);(p/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
