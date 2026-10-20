"""Native GDN beta distributions and beta<=1 inference interventions."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from mapping.checkpoint import load_checkpoint
from mapping.loader import load_banks
from mapping.data import make_batch
from mapping.tasks import config_and_spec
from mapping.models.se2d.hybrid4 import DECAY_TAU
from experiments.hybrid6.mechanistic.run import tensor,queries
from core import native

parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='results/se2d/hybrid6_causal/s/local_causal_inert_saw_s_100k_ieee_seed0/final.pt');parser.add_argument('--out',default='results/hybrid6_mechanistic_beta');opts=parser.parse_args()
out=Path(opts.out);out.mkdir(parents=True,exist_ok=True);ck=opts.checkpoint
torch.set_num_threads(4);model,args,spec,task=load_checkpoint(ck,'cuda')
cfg,_=config_and_spec('se2d',32,3,1807,'uniform',9,0.,'mnist','saw',spec.offset_max);_,bank=load_banks('data/mnist.npz',cfg);rng=np.random.default_rng(1807)
values={j:{'observation':[],'odometry':[]} for j in (0,2)};decays={j:{'observation':[],'odometry':[]} for j in (0,2)};query_max=[];rows=[];symbols=[]
with torch.no_grad():
 for i in range(16):
  b=queries(tensor(make_batch(rng,cfg,bank,8)),[8,16,24],32);symbols.append(b['odometry'].flatten());target=b['observations'][:,[8,16,24]].float()/255;hooks=[];originals=[];masks={}
  for j in (0,2):
   m=model.blocks[j].mixer;originals.append((m,m.scan))
   def pre(m,a,j=j):
    x,layout=a;h=layout.gather(x);beta=2*torch.sigmoid(m.beta(h).float());decay=torch.exp(torch.nn.functional.logsigmoid(m.decay(h).float())/DECAY_TAU);ix=torch.arange(h.shape[1],device=h.device)%10==9
    for name,mask in [('observation',~ix),('odometry',ix)]:values[j][name].append(beta[:,mask].flatten(0,1).cpu());decays[j][name].append(decay[:,mask].flatten(0,1).cpu())
    masks[j]=layout.hist
   hooks.append(m.register_forward_pre_hook(pre));old=m.scan
   def scan(q,k,v,g,beta,j=j,old=old):
    query_max.append(float(beta[~masks[j]].abs().max()));return old(q,k,v,g,beta)
   m.scan=scan
  try:y=native(model,b)
  finally:
   for h in hooks:h.remove()
   for m,old in originals:m.scan=old
  row={'native':float((y-target).square().mean())}
  for label,layers in [('clamp_gdn0',(0,)),('clamp_gdn2',(2,)),('clamp_both',(0,2))]:
   restores=[]
   for j in layers:
    m=model.blocks[j].mixer;old=m.scan;restores.append((m,old))
    def clamp(q,k,v,g,beta,old=old):return old(q,k,v,g,beta.clamp(max=1))
    m.scan=clamp
   try:yy=native(model,b);row[label]=float((yy-target).square().mean())
   finally:
    for m,old in restores:m.scan=old
  rows.append(row)
result={'checkpoint':ck,'seed':1807,'episodes':128,'effective_query_beta_abs_max':max(query_max),'native_and_clamps':rows,'layers':{}}
for j in (0,2):
 result['layers'][j]={}
 for kind in values[j]:
  v=torch.cat(values[j][kind]);d=torch.cat(decays[j][kind]);result['layers'][j][kind]=[{'head':h,'n':len(v),'mean':float(v[:,h].mean()),'quantiles':torch.quantile(v[:,h],torch.tensor([0.,.05,.5,.95,1.])).tolist(),'fraction_gt1':float((v[:,h]>1).float().mean()),'fraction_gt1_9':float((v[:,h]>1.9).float().mean()),'fraction_gt1_99':float((v[:,h]>1.99).float().mean()),'mean_decay_multiplier':float(d[:,h].mean())} for h in range(4)]
result['odometry_mean_beta_by_turn']={str(j):[torch.cat(values[j]['odometry'])[torch.cat(symbols)//4==t].mean(0).tolist() for t in range(4)] for j in (0,2)}
(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['layers'],indent=2));print('clamps',{k:float(np.mean([r[k] for r in rows])) for k in rows[0]})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,4,figsize=(12,5),sharex=True,sharey=True)
for r,j in enumerate((0,2)):
 for h in range(4):
  ax=axes[r,h]
  for kind in values[j]:ax.hist(torch.cat(values[j][kind])[:,h].numpy(),bins=np.linspace(0,2,51),density=True,histtype='step',label=kind)
  ax.axvline(1,color='gray',linestyle='--');ax.set_title(f'GDN {j}, head {h}');ax.set_xlabel('beta')
axes[0,0].legend();fig.suptitle('Causal inert S: history beta distributions (128 fresh episodes)');fig.tight_layout();fig.savefig(out/'beta.png',dpi=160)
