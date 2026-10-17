"""Eval mse (true | odometry shuffled) of two runs at matched steps.

    python experiments/se2d_saw/compare_logs.py REF_DIR RUN_DIR
"""
import json
import os
import sys


def evals(d):
    out = {}
    for line in open(os.path.join(d, 'log.jsonl')):
        r = json.loads(line)
        if 'eval' in r:
            out[r['step']] = (r['eval']['mse'], r['eval_shuffled']['mse'],
                              r['eval'].get('mse_t24_32'))
    return out


a, b = evals(sys.argv[1]), evals(sys.argv[2])
print(f'{"step":>7} {"ref mse":>9} {"ref shuf":>9} {"ref t24+":>9} | '
      f'{"run mse":>9} {"run shuf":>9} {"run t24+":>9}')
for s in sorted(set(a) & set(b)):
    print(f'{s:7d} ' + ' '.join(f'{x:9.4f}' for x in a[s]) + ' | '
          + ' '.join(f'{x:9.4f}' for x in b[s]))
