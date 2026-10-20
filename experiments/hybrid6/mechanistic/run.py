"""Hybrid6 S mechanistic suite adapted from the Hybrid4 transfer screen. No training, no modifications to checkpoints."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from mapping.checkpoint import load_checkpoint
from mapping.loader import load_banks
from mapping.data import make_batch,render_map,crop_windows
from mapping.tasks import config_and_spec
from mapping import se2d
from mapping.models.se2d.hybrid4 import apply_rope,rope_cos_sin
from core import native,capture,replay,query_input,metrics
from report import write_report


def tensor(b):return {k:torch.from_numpy(v) for k,v in b.items() if isinstance(v,np.ndarray)}
def queries(b,sources,time):
 b=dict(b);pos,hd=b['positions'],b['headings'];b['query_pose']=torch.from_numpy(se2d.query_pose(pos[:,sources].numpy(),hd[:,sources].numpy(),pos[:,time:time+1].numpy(),hd[:,time:time+1].numpy()));b['query_time']=torch.full((len(pos),len(sources)),time,dtype=torch.long);return b

def readout(model,b):
 T=b['observations'].shape[1]-1;src=[T//4,T//2,3*T//4];b=queries(b,src,T);target=b['observations'][:,src].float()/255;mem,cap=capture(model,b);original=native(model,b);pred,prob,q=replay(model,mem);err=float((pred-original).square().mean());qerr=float((q-mem['native_q']).norm()/mem['native_q'].norm().clamp_min(1e-12));assert np.isfinite(err) and np.isfinite(qerr),(err,qerr)
 out={'replay_quality_pass':err<1e-4 and qerr<.03,'native':metrics(original,target),'replay':metrics(pred,target,prob,src),'replay_mse_vs_native':err,'capture_mse_vs_native':float((cap-original).square().mean()),'query_relative_error':qerr,'ablations':{}}
 configs={'final_history_only':dict(final_only=True),'zero_gdn0':dict(zero_states=(0,)),'zero_gdn2':dict(zero_states=(2,)),'zero_both':dict(zero_states=(0,2)),'final_only_zero_both':dict(final_only=True,zero_states=(0,2)),'skip_final_mlp':dict(skip_mlp=3)}
 configs.update({f'zero_layer{j}_head{h}':dict(zero_head=(j,h)) for j in [0,2,3] for h in range(4)})
 for name,kw in configs.items():y,p,_=replay(model,mem,**kw);out['ablations'][name]=metrics(y,target,p,src)
 donor=dict(b);donor['observations']=b['observations'].clone();donor['observations'][:,src[1]]=b['observations'][:,src[1]].roll(1,0);dm,_=capture(model,donor);newtarget=donor['observations'][:,src].float()/255;contrast=(newtarget[:,1]-target[:,1]).square().mean((-1,-2));eligible=contrast>.005;out['texture_contrast_eligible_fraction']=float(eligible.float().mean());out['texture_swaps']={};lo=10*src[1];hi=lo+9
 for name in ['values','keys','states']+[f'value_head{h}' for h in range(4)]:
  m=dict(mem);m['kv']=dict(mem['kv']);k,v=[x.clone() for x in mem['kv'][3]]
  if name=='keys':k[:,lo:hi]=dm['kv'][3][0][:,lo:hi]
  elif name=='states':m['states']=dm['states']
  elif name=='values':v[:,lo:hi]=dm['kv'][3][1][:,lo:hi]
  else:h=int(name[-1]);v[:,lo:hi,h]=dm['kv'][3][1][:,lo:hi,h]
  m['kv'][3]=(k,v);y,p,_=replay(model,m);row=metrics(y,target,p,src);row['donor_texture_mse']=float((y[:,1]-newtarget[:,1]).square().mean());row['unmodified_queries_pixel_mse']=float((y[:,[0,2]]-target[:,[0,2]]).square().mean())
  if eligible.any():
   delta=newtarget[eligible,1]-target[eligible,1];row['texture_projection']=float(((y[eligible,1]-pred[eligible,1])*delta).sum()/delta.square().sum())
  else:row['texture_projection']=None
  out['texture_swaps'][name]=row
 return out

def rewind(model,b):
 T=b['observations'].shape[1]-1;u=T//2;src=[0,T//8,T//4];b=queries(b,src,T);mem,nat=capture(model,b);target=b['observations'][:,src].float()/255;early=queries(b,src,u);early['observations']=b['observations'][:,:u+1];early['odometry']=b['odometry'][:,:u];dm,_=capture(model,early);out={}
 for name,layers in [('positive_current_frame',None),('earlier_query_no_rewind',()),('rewind_gdn0',(0,)),('rewind_gdn2',(2,)),('rewind_both',(0,2))]:
  m=dict(mem);m['states']=dict(mem['states'])
  if layers is not None:
   m['input']=query_input(model,early['query_pose'])
   for j in layers:m['states'][j]=dm['states'][j]
  y,p,_=replay(model,m,final_only=True);out[name]=metrics(y,target,p,src)
  y,p,_=replay(model,m,final_only=False);out['all_history_'+name]=metrics(y,target,p,src)
 return out

def translation(model,b):
 T=b['observations'].shape[1]-1;cut=T//2-1;src=[0,T//4,3*T//4];b=queries(b,src,T);odo=b['odometry'].numpy().copy();odo[:,cut]=4*(odo[:,cut]//4)+(odo[:,cut]%4+2)%4;hd=b['headings'].numpy();pp,hh=se2d.compose(odo,9,hd[:,0]);clean=np.array([len(np.unique(x,axis=0))==T+1 for x in pp]);out={'collision_free_episodes':int(clean.sum())}
 if not clean.any():return out
 b={k:v[clean] for k,v in b.items()};odo=torch.from_numpy(odo[clean]);pp=pp[clean];hh=hh[clean];comp=torch.from_numpy(se2d.query_pose(pp[:,src],hh[:,src],pp[:,-1,None],hh[:,-1,None])).to(b['query_pose'].dtype);wrong=b['query_pose'].clone();wrong[...,:2]=2*b['query_pose'][...,:2]-comp[...,:2];target=b['observations'][:,src].float()/255
 for name,o,qp in [('baseline',b['odometry'],b['query_pose']),('edited_unshifted',odo,b['query_pose']),('edited_compensated',odo,comp),('wrong_sign',odo,wrong)]:
  bb=dict(b);bb['odometry']=o;bb['query_pose']=qp;y=native(model,bb);out[name]={'earlier_source_mse':float((y[:,:2]-target[:,:2]).square().mean()),'later_source_mse':float((y[:,2]-target[:,2]).square().mean())}
 return out

def route(kind):return np.array([(y,x) for y in range(4) for x in (range(8) if y%2==0 else range(7,-1,-1))]) if kind=='row' else np.array([(y,x) for x in range(8) for y in (range(4) if x%2==0 else range(3,-1,-1))])
def cosine(x,y):return float(np.mean((x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)+1e-12)))
def grid(model,bank,seed,B):
 rng=np.random.default_rng(seed);worlds=[(render_map(bank,rng.integers(0,len(bank),(3,4)),28,np.zeros((3,4),np.int8)),rng.integers(0,12,2)) for _ in range(B)];mems={};data={};out={'native_mse':{},'key_geometry':{},'key_swaps':{},'query_lag_cosines':{}}
 for kind in ['row','column']:
  pos=np.stack([route(kind)*9+phase for _,phase in worlds]).astype(np.int32);obs=np.stack([crop_windows(im,pp,9) for (im,_),pp in zip(worlds,pos)]);heads=np.zeros((B,32),np.int64);dv=(pos[:,1:]-pos[:,:-1])//9;dirs=(dv[:,:,None]==np.array([[-1,0],[0,1],[1,0],[0,-1]])[None,None]).all(-1).argmax(-1);b=tensor({'observations':obs,'odometry':se2d.encode(dirs,heads),'positions':pos,'headings':heads});b=queries(b,[0,16,31],31);mem,pred=capture(model,b);mems[kind]=mem;data[kind]=b;out['native_mse'][kind]=float((pred-b['observations'][:,[0,16,31]].float()/255).square().mean());k=mem['raw_k'][:,np.arange(32)*10+4].float().cpu().numpy();dist=np.linalg.norm(route(kind)[:,None]-route(kind)[None],axis=-1);lag=np.abs(np.arange(32)[:,None]-np.arange(32)[None]);ij=np.triu_indices(32,1);rows=[]
  for h in range(4):
   z=k[:,:,h];z=z/np.linalg.norm(z,axis=-1,keepdims=True);d=1-np.einsum('bid,bjd->bij',z,z);rows.append({'head':h,'spatial_distance_correlation':float(np.mean([np.corrcoef(v[ij],dist[ij])[0,1] for v in d])),'time_lag_correlation':float(np.mean([np.corrcoef(v[ij],lag[ij])[0,1] for v in d]))})
  out['key_geometry'][kind]=rows
  line=np.array([[9,x] for x in range(64)]);qp=np.concatenate([np.broadcast_to(line-route(kind)[-1]*9,(B,64,2)),np.zeros((B,64,1),np.int64)],-1);mm=dict(mem);mm['input']=query_input(model,torch.from_numpy(qp));_,_,q=replay(model,mm);q=q[:,:,4].numpy();out['query_lag_cosines'][kind]={str(l):[cosine(q[:,:64-l,h],q[:,l:,h]) for h in range(4)] for l in [1,9,18,27,36,54]}
 row,col=route('row'),route('column');perm=np.array([np.flatnonzero((col==v).all(-1))[0] for v in row]);ix=np.array([t*10+j for t in range(32) for j in range(9)]);dx=np.array([perm[t]*10+j for t in range(32) for j in range(9)]);mem=mems['row'];raw=mem['raw_k'].clone();raw[:,ix]=mems['column']['raw_k'][:,dx];c,s=rope_cos_sin((torch.arange(319,device='cuda')//10)[None].expand(B,-1),32)
 with torch.autocast('cuda',dtype=torch.bfloat16):swapped=apply_rope(raw,c,s)
 b=queries(data['row'],list(range(32)),31);target=b['observations'].float()/255;qref=None
 for name,keys in [('intact',mem['kv'][3][0]),('same_cell_donor_keys_recipient_rope',swapped)]:
  mm=dict(mem);mm['input']=query_input(model,b['query_pose']);mm['kv']=dict(mem['kv']);mm['kv'][3]=(keys,mem['kv'][3][1]);y,p,q=replay(model,mm)
  if qref is None:qref=q
  else:assert torch.equal(q,qref)
  out['key_swaps'][name]=metrics(y,target,p,list(range(32)))
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',action='append',required=True,help='LABEL=PATH; repeat for paired comparison');ap.add_argument('--steps',type=int,choices=[32,64],required=True);ap.add_argument('--episodes',type=int,default=32);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--grid-worlds',type=int,default=8);ap.add_argument('--seed',type=int,default=1801101);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--quality-mse',type=float,default=.005);args=ap.parse_args();assert args.episodes>=args.batch>=2 and args.episodes%args.batch==0 and args.grid_worlds>=2
 args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);result={'schema':1,'steps':args.steps,'episodes':args.episodes,'seed':args.seed,'grid_worlds':args.grid_worlds,'quality_mse_threshold':args.quality_mse,'notes':['All heads reported; no assumption that head identities transfer.','Quality readiness is a heuristic interpretation gate, not a statistical pass/fail.','All checkpoints receive identical generated episodes at the explicitly selected horizon.','Translation counterfactuals report only collision-free paths; small cohorts are inconclusive.','Texture donor intervention is not a globally consistent physical episode.','Grid lag periodicity is descriptive, not proof of a phase mechanism.'],'models':{}}
 result['runtime']={'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),'precision':'bf16 autocast; state capture matches checkpoint scan (IEEE FP32 chunk32 or TF32 native chunk64); FP32 state read; SDPA replay attention','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
 for item in args.checkpoint:
  label,path=item.split('=',1);assert label not in result['models'];start=time.time();model,meta,spec,task=load_checkpoint(path,'cuda');assert task=='se2d' and model.query_mode=='inert' and len(model.blocks)==4 and model.width==128
  for p in model.parameters():p.requires_grad_(False)
  cfg,_=config_and_spec('se2d',args.steps,3,args.seed,'uniform',9,0.,'mnist','saw',spec.offset_max);_,bank=load_banks('data/mnist.npz',cfg);rng=np.random.default_rng(args.seed);rows=[]
  for _ in range(args.episodes//args.batch):
   b=tensor(make_batch(rng,cfg,bank,args.batch));rows.append({'readout':readout(model,b),'rewind':rewind(model,b),'translation':translation(model,b)})
  native_mse=float(np.mean([r['readout']['native']['pixel_mse'] for r in rows]));entry={'checkpoint':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'scan_precision':getattr(model,'scan_precision','tf32'),'checkpoint_training_horizon':meta.get('episode_steps'),'sampled_native_mse':native_mse,'quality_ready':native_mse<=args.quality_mse,'batches':rows,'grid':grid(model,bank,args.seed+1,args.grid_worlds),'elapsed_s':time.time()-start};result['models'][label]=entry;(args.out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'model':label,'quality_ready':entry['quality_ready'],'native_mse':native_mse,'seconds':entry['elapsed_s']}),flush=True);del model;torch.cuda.empty_cache()
 if len(result['models'])>1:
  result['paired_quality_ready']=all(v['quality_ready'] for v in result['models'].values());(args.out/'results.json').write_text(json.dumps(result,indent=2))
 write_report(result,args.out/'report.md')
if __name__=='__main__':main()
