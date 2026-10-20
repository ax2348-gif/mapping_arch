"""Final S renders plus translation/blur diagnostics (not pose-error estimates)."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mapping.checkpoint import load_checkpoint
from mapping import task_se2c as task

torch.set_num_threads(2)
root=Path('results/se2c/hybrid7c/s/from_se2d100k_agaga_s_100k_ieee_seed0')
m,args,spec,_=load_checkpoint(root/'final.pt','cuda');data=task.data(args,'cuda',eval_batches=16,batch=8)
ys=[];ts=[];ks=[];times=[];poses=[]
with torch.no_grad():
 for b in data['held_out']:
  obs,odo,qp,target,qt=task.prepare(b,'cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):y=m(obs,odo,qp,qt)
  ys.append(y.float().cpu().flatten(0,1));ts.append(target.cpu().flatten(0,1));ks.append(task.query_kind(qp,b['stats']['found']).cpu().flatten());times.append(qt.cpu().flatten());poses.append(qp.cpu().flatten(0,1))
y=torch.cat(ys);target=torch.cat(ts);kind=torch.cat(ks);qt=torch.cat(times);qp=torch.cat(poses)
# Compare a fixed central9 region so every candidate +/-2px translation has support.
z=y[:,None];truth=target[:,3:12,3:12];base=((y[:,3:12,3:12]-truth)**2).mean((1,2))
r=torch.arange(3,12);gy,gx=torch.meshgrid(r,r,indexing='ij');best=base.clone();bestxy=torch.zeros(len(y),2)
for dy in np.arange(-2,2.01,.5):
 for dx in np.arange(-2,2.01,.5):
  grid=torch.stack(((gx+dx)/7-1,(gy+dy)/7-1),-1).float()[None].expand(len(y),-1,-1,-1)
  shifted=F.grid_sample(z,grid,align_corners=True)[:,0];err=((shifted-truth)**2).mean((1,2));win=err<best;best[win]=err[win];bestxy[win]=torch.tensor([dy,dx]).float()
# Does a blurred target explain the prediction better than the sharp target?
blurfit=base.clone();sigmas=torch.zeros(len(y))
for sigma in [.5,1.,1.5,2.]:
 r=torch.arange(-6,7).float();g=torch.exp(-r*r/(2*sigma*sigma));g/=g.sum();kernel=(g[:,None]*g[None,:])[None,None]
 blurred=F.conv2d(F.pad(target[:,None],(6,6,6,6),mode='reflect'),kernel)[:,0]
 err=((y[:,3:12,3:12]-blurred[:,3:12,3:12])**2).mean((1,2));win=err<blurfit;blurfit[win]=err[win];sigmas[win]=sigma
mse=((y-target)**2).mean((1,2));out=root/'geometry_diagnostic';out.mkdir(exist_ok=True)
summary={}
for k,name in enumerate(['fallback','near','mid','far']):
 mask=kind==k
 summary[name]={'queries':int(mask.sum()),'full_mse':float(mse[mask].mean()),'center_mse':float(base[mask].mean()),'oracle_translated_prediction_center_mse':float(best[mask].mean()),'oracle_blurred_target_center_mse':float(blurfit[mask].mean()),'median_best_shift_px':float(bestxy[mask].norm(dim=1).median()),'median_target_blur_sigma':float(sigmas[mask].median())}
# Median and 90th-percentile error cases within each kind, explicitly selected.
selected=[]
for k in range(4):
 ids=(kind==k).nonzero().flatten();order=ids[mse[ids].argsort()]
 for q in [.5,.9]:selected.append((int(order[round(q*(len(order)-1))]),k,q))
fig,axes=plt.subplots(8,4,figsize=(10,19),layout='constrained')
for row,(i,k,quant) in enumerate(selected):
 for col,a in enumerate([target[i],y[i],y[i],(y[i]-target[i]).abs()]):
  ax=axes[row,col];ax.imshow(a,vmin=0,vmax=1,cmap='gray',interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
  if col==2:
   ax.contour(target[i],levels=[.5],colors=['cyan'],linewidths=1)
   ax.contour(y[i],levels=[.5],colors=['magenta'],linewidths=1)
  if row==0:ax.set_title(['Target','Prediction','Contours: target cyan / pred magenta','Absolute error'][col],fontsize=9)
 axes[row,0].set_ylabel(f'{["fallback","near","mid","far"][k]} p{int(quant*100)}\nt={int(qt[i])}, MSE={float(mse[i]):.3f}',fontsize=9)
fig.suptitle('Hybrid7c S final100k | selected median and high-error cases per query kind\n15x15 pixels enlarged without smoothing; contour threshold0.5',fontsize=12)
fig.savefig(out/'comparisons.png',dpi=140);plt.close(fig)
np.savez_compressed(out/'arrays.npz',target=target.numpy(),prediction=y.numpy(),kind=kind.numpy(),query_time=qt.numpy(),query_pose=qp.numpy(),selected=np.array(selected))
(out/'metrics.json').write_text(json.dumps({'episodes':128,'queries':len(y),'notes':'Oracle per-query fits on fixed central9 crop. Best shifts in +/-2px at0.5px increments; not decoded pose error. Blurring target measures smoothing compatibility, not recovered accuracy. Fits overlap and cannot partition causes.','metrics':summary},indent=2));print(json.dumps(summary,indent=2))
