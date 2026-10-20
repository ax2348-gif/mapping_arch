"""Strict adapter transfer and zero-shot SE2C evaluation before training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import run,train
from mapping import task_se2c as task
from mapping.checkpoint import load_checkpoint
ck='results/se2d/hybrid7/s/hybrid7_causal_inert_saw_s_agaga_100k_ieee_seed0/final.pt'
torch.set_num_threads(4)
cfg=run.resolved(run.load(['+experiment=hybrid7_se2c_transfer',f'run.init_from={ck}']))
e=run.build(cfg);init=train.load_init(e.run,e,'se2c',cfg)
d=task.from_config(e.task.generator.config(e.task,0),torch.device('cuda'),eval_batches=16,batch=64)
m=e.model.build(d['spec']);m.load_state_dict(init['state'],strict=True)
assert all(torch.equal(v,init['state'][k]) for k,v in m.state_dict().items())
m=m.cuda();ac=torch.autocast('cuda',dtype=torch.bfloat16)
from mapping.tasks import get
metrics=task.metrics
result={'checkpoint':ck,'sha256':init['sha256'],'all_weights_exact':True,'parameters':sum(p.numel() for p in m.parameters()),'episodes':1024}
for shuffled in (False,True):
 result['shuffled' if shuffled else 'intact']=train.evaluate(m,d['held_out'],torch.device('cuda'),ac,task.prepare,metrics,True,32,shuffled,by_query=task.metrics_by_query)
p=Path('results/hybrid7_se2c_transfer_preflight');p.mkdir(exist_ok=True);(p/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
