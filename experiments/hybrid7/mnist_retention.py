"""Frozen pre/post-transfer MNIST evaluation; no training process changes."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import run,train
from mapping.checkpoint import load_checkpoint
from mapping.tasks import config_and_spec
from mapping.loader import load_banks,eval_set
from mapping import task_se2d as td,task_se2c as tc
from mapping.models.se2d.hybrid7 import Hybrid7
from mapping.models.se2c.hybrid7c import Hybrid7c

torch.set_num_threads(2)
ck=Path(sys.argv[1]);parent=Path('results/se2d/hybrid7/s/hybrid7_causal_inert_saw_s_agaga_100k_ieee_seed0/final.pt')
a=torch.load(parent,map_location='cpu',weights_only=False);b=torch.load(ck,map_location='cpu',weights_only=False)
kw=a['arch']['kwargs'];assert b['arch']['kwargs']==kw
cfg,spec=config_and_spec('se2d',32,8,0,'uniform',9,0.,'mnist','saw',288)
_,bank=load_banks('data/mnist.npz',cfg);discrete=eval_set(cfg,bank,8,16,0)
e=run.build(run.resolved(run.load(['+experiment=hybrid7_se2c_transfer','task.generator.texture=mnist'])))
d=tc.from_config(e.task.generator.config(e.task,0),torch.device('cuda'),eval_batches=16,batch=8)
result={'checkpoint':str(ck),'sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'source_step':b['step'],'parent':str(parent),'episodes_per_task':128,'seed':0,'conditions':{}}
for name,cls,sp,task,batches in [('se2d_mnist9',Hybrid7,spec,td,discrete),('se2c_mnist15',Hybrid7c,d['spec'],tc,d['held_out'])]:
 result['conditions'][name]={}
 for label,state in [('before',a['model']),('after',b['model'])]:
  model=cls(sp,**kw);model.load_state_dict(state,strict=True);model=model.cuda()
  row={}
  for shuffled in (False,True):
   row['shuffled' if shuffled else 'intact']=train.evaluate(model,batches,torch.device('cuda'),torch.autocast('cuda',dtype=torch.bfloat16),task.prepare,task.metrics,True,32,shuffled,by_query=getattr(task,'metrics_by_query',None))
  result['conditions'][name][label]=row
  print(name,label,row['intact']['mse'],row['shuffled']['mse'],flush=True)
  del model
out=Path('results/hybrid7_mnist_retention')/ck.stem;out.mkdir(parents=True,exist_ok=False);(out/'results.json').write_text(json.dumps(result,indent=2))
