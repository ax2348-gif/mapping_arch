"""GPU kernel/gradient/checkpoint and permutation smoke, not a learning run."""
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import run
from mapping.checkpoint import build_from_arch
from mapping.hybrid6_geometry import patchify,patch_calibration,unpatchify
from mapping.models.se2d.hybrid4 import GatedDeltaNet

torch.set_num_threads(4);torch.manual_seed(0)
e=run.build(run.resolved(run.load(['+experiment=hybrid7_initial'])))
_,spec=e.task.generator.data(e.task,0);m=e.model.build(spec).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
B,F,Q=2,33,8
obs=torch.rand(B,F,9,9,device='cuda');odo=torch.randint(16,(B,F-1),device='cuda');qp=torch.cat((torch.randint(-70,71,(B,Q,2),device='cuda'),torch.randint(4,(B,Q,1),device='cuda')),-1);qt=torch.randint(F,(B,Q),device='cuda');target=torch.rand(B,Q,9,9,device='cuda');losses=[]
for _ in range(3):
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=(m(obs,odo,qp,qt).float()-target).square().mean()
 assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step();losses.append(loss.item())
for name in ('registers','pose_embed','sensor_embed','patch_embed'):
 g=getattr(m,name).weight.grad;assert g is not None and g.abs().sum()>0
restored=build_from_arch(e.model.record(),spec).cuda();restored.load_state_dict(m.state_dict(),strict=True)
checks={}
with torch.no_grad():
 for dtype in (torch.float32,torch.bfloat16):
  ctx=lambda:torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16)
  trans,queries,known=m._task_poses(odo,qp);pixels=patchify(obs);poses=torch.eye(4,device='cuda').expand(B,F,9,4,4);cal=patch_calibration(9,9,'cuda').expand(B,F,9,5);qcal=patch_calibration(9,9,'cuda').expand(B,Q,9,5);queries=queries[:,:,None].expand(B,Q,9,4,4);perm=torch.randperm(9,device='cuda')
  with ctx():
   base=m(obs,odo,qp,qt);same=restored(obs,odo,qp,qt)
   y=m.forward_cameras(pixels[:,:,perm],poses[:,:,perm],cal[:,:,perm],trans,queries,qcal,qt,transition_known=known)
  assert torch.equal(base,same)
  err=(base.float()-unpatchify(y,9,9).float());checks[str(dtype)]={'permutation_max_abs':err.abs().max().item(),'permutation_mse':err.square().mean().item()}
  assert err.square().mean()< (1e-10 if dtype==torch.float32 else 1e-7),checks
 # Runtime larger grids + backward already checked on CPU.
 with torch.autocast('cuda',dtype=torch.bfloat16):large=m(torch.rand(B,F,15,15,device='cuda'),odo,qp,qt)
 assert large.shape==(B,Q,15,15) and torch.isfinite(large).all()
# Count register updates and convolution positions at real T32.
seen=[];hooks=[]
for block in m.blocks:
 if isinstance(block.mixer,GatedDeltaNet):hooks.append(block.mixer.short.register_forward_pre_hook(lambda mod,a:seen.append(a[0].shape[-1])))
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):m(obs,odo,qp,qt)
for h in hooks:h.remove()
assert seen==[33*5+3]*sum(isinstance(b.mixer,GatedDeltaNet) for b in m.blocks)
out=Path('results/hybrid7_s_agaga_smoke');out.mkdir(exist_ok=True)
result={'status':'passed','parameters':sum(p.numel() for p in m.parameters()),'device':torch.cuda.get_device_name(),'synthetic_losses':losses,'checks':checks,'checkpoint_roundtrip_exact':True,'conv_lengths_including_padding':seen,'large_output_shape':list(large.shape),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('mapping/models/se2d/hybrid7.py'),Path('mapping/models/se2c/hybrid7c.py'))}}
(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
