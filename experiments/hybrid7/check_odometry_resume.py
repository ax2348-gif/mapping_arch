"""CPU-only preflight of a real AGAGA checkpoint; never launches training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import run
from mapping.checkpoint import load_checkpoint
from train import validate_branch_transition,branch_changes

torch.set_num_threads(2)
p=Path(sys.argv[1]).resolve();ck=torch.load(p,map_location='cpu',weights_only=False)
cfg=run.resolved(run.load(['+experiment=hybrid7_odometry',f'run.branch={p}','run.branch_transition=hybrid7_odometry_writes','run.name=odometry_resume_preflight']))
e=run.build(cfg)
source,args,spec,task=load_checkpoint(p,'cpu');target=e.model.build(spec)
from types import SimpleNamespace
validate_branch_transition(ck,e.model.record(),'hybrid7_odometry_writes',SimpleNamespace(model='hybrid7_odom',task='se2d'))
changed=branch_changes(p.parent/'config.yaml',cfg)
assert set(changed)<= {'model.name','model.arch.odometry_writes'},changed
target.load_state_dict(ck['model'],strict=True)
assert [(n,p.shape) for n,p in source.named_parameters()]==[(n,p.shape) for n,p in target.named_parameters()]
decay=[];no_decay=[]
for n,v in target.named_parameters():(decay if v.ndim>=2 else no_decay).append(v)
opt=torch.optim.AdamW([{'params':decay,'weight_decay':.1},{'params':no_decay,'weight_decay':0.}],lr=3e-4,betas=(.9,.95));opt.load_state_dict(ck['opt'])
for param,state in opt.state.items():
 for key in ('exp_avg','exp_avg_sq'):assert state[key].shape==param.shape
out={'checkpoint':str(p),'source_step':ck['step'],'source_run':ck.get('provenance'),'parameters':sum(v.numel() for v in target.parameters()),'strict_full_weights':True,'optimizer_loaded':True,'optimizer_parameter_states':len(opt.state),'config_changes':changed,'target_arch':e.model.record(),'training_launched':False}
dest=Path('results/hybrid7_odometry_preflight');dest.mkdir(exist_ok=True);(dest/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
