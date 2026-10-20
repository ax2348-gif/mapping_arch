"""Durable, at-most-once M launch after successful local S completion."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT=ROOT/'results/hybrid7_se2c_medium_queue';OUT.mkdir(exist_ok=True)
lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PARENT=ROOT/'results/se2c/hybrid7c/s/from_se2d100k_agaga_s_100k_ieee_seed0'
EXIT=ROOT/'results/hybrid7_se2c_launch/exit_code'
TARGET=ROOT/'results/se2c/hybrid7c/m/fresh_agagaga_m_100k_ieee_seed0'
MARK=OUT/'launched.json'
def status(**kw):
 kw['updated_unix']=time.time();tmp=OUT/'status.tmp';tmp.write_text(json.dumps(kw,indent=2));tmp.replace(OUT/'status.json');print(json.dumps(kw),flush=True)
if MARK.exists() or TARGET.exists():
 status(state='refused_duplicate',reason='launch marker or target exists');raise SystemExit(0)
status(state='waiting',parent=str(PARENT),target=str(TARGET))
while True:
 if EXIT.exists():
  code=int(EXIT.read_text().strip())
  if code!=0:
   status(state='blocked_parent_failure',exit_code=code);raise SystemExit(1)
  final=PARENT/'final.json'
  if final.exists() and (PARENT/'final.pt').exists():
   if json.loads(final.read_text()).get('steps')!=100000:
    status(state='blocked_unexpected_parent_steps');raise SystemExit(1)
   break
 time.sleep(30)
cmd=['/home/adrianna/miniconda3/bin/python','-u','run.py','+experiment=hybrid7_se2c_medium','run.note=User queued medium register-only Hybrid7c continuous causal inert run after S transfer completes with fresh weights and optimizer']
with MARK.open('x') as f:json.dump({'command':cmd,'time':time.time()},f)
status(state='starting',command=cmd)
env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
with (OUT/'training.log').open('x') as log:
 proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 status(state='running',pid=proc.pid,target=str(TARGET));code=proc.wait()
status(state='completed' if code==0 else 'failed',exit_code=code,target=str(TARGET))
