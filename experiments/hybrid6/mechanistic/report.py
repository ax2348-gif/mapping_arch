"""Readable summaries; batch spread is descriptive, not a confidence interval."""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def number(x):
    return 'NA' if x is None else f'{x:.5g}'


def summarize(values):
    values = [v for v in values if v is not None]
    return 'NA' if not values else f'{mean(values):.5g} ± {pstdev(values):.2g}'


def write_report(result, destination):
    lines = ['# Mechanistic transfer screen', '',
             f"Horizon: {result['steps']}; episodes/model: {result['episodes']}; grid worlds: {result['grid_worlds']}; seed: {result['seed']}.", '',
             'MSE uses pixels scaled to [0,1]. ± denotes batch standard deviation, not uncertainty of the mean. All checkpoints see the same episodes. This small screen identifies follow-up experiments; it does not establish equivalence.', '']
    for label, model in result['models'].items():
        batches = model['batches']
        if any(not b['readout'].get('replay_quality_pass',True) for b in batches):
            lines += ['**Replay tolerance exceeded for '+label+': small intervention effects are inconclusive. Check the discrepancies below.**', '']
        lines += [f'## {label}', '',
                  f"Checkpoint: `{model['checkpoint']}` (training horizon {model['checkpoint_training_horizon']}).", '',
                  f"Aligned native MSE: {model['sampled_native_mse']:.5g}. Quality heuristic: {'ready' if model['quality_ready'] else 'UNDERTRAINED / POOR RENDERING — mechanistic failures inconclusive'}. Grid rendering must also be checked separately below.", '',
                  f"Maximum replay/native MSE: {max(b['readout']['replay_mse_vs_native'] for b in batches):.5g}; maximum relative final-Q discrepancy: {max(b['readout']['query_relative_error'] for b in batches):.5g}.", '',
                  '| Readout intervention | Pixel MSE | Change from intact replay |',
                  '|---|---:|---:|']
        for name in ['replay'] + list(batches[0]['readout']['ablations']):
            vals = [(b['readout']['replay'] if name == 'replay' else b['readout']['ablations'][name])['pixel_mse'] for b in batches]
            diffs = [v-b['readout']['replay']['pixel_mse'] for v,b in zip(vals,batches)]
            lines.append(f'| {name} | {summarize(vals)} | {summarize(diffs)} |')
        lines += ['', '| Final attention head | Intact correct-patch argmax |', '|---|---:|']
        for h in range(4):
            lines.append(f"| {h} | {summarize([b['readout']['replay']['patch_accuracy_by_head'][h] for b in batches])} |")
        lines += ['', 'Texture projection: 0 = original texture, 1 = donor displacement along the original-to-donor image difference. Only contrasting donors contribute; values outside [0,1] are possible.', '',
                  f"Eligible donor fraction: {summarize([b['readout']['texture_contrast_eligible_fraction'] for b in batches])}.", '',
                  '| Swap | Texture projection | Donor-image MSE | Other queries MSE |', '|---|---:|---:|---:|']
        for name in batches[0]['readout']['texture_swaps']:
            rows = [b['readout']['texture_swaps'][name] for b in batches]
            lines.append('| '+name+' | '+' | '.join(summarize([r[k] for r in rows]) for k in ['texture_projection','donor_texture_mse','unmodified_queries_pixel_mse'])+' |')
        lines += ['', '| Rewind intervention (final cache retained) | Pixel MSE |', '|---|---:|']
        for name in batches[0]['rewind']:
            lines.append(f"| {name} | {summarize([b['rewind'][name]['pixel_mse'] for b in batches])} |")
        rows = [b['translation'] for b in batches if b['translation']['collision_free_episodes']]
        n = sum(r['collision_free_episodes'] for r in rows)
        lines += ['', f'Translation reversal: {n} collision-free episodes. Conditional on this selected subset; small counts are inconclusive.', '',
                  '| Intervention | Earlier sources MSE | Later source MSE |', '|---|---:|---:|']
        if n:
            for name in ['baseline','edited_unshifted','edited_compensated','wrong_sign']:
                vals = [sum(r[name][k]*r['collision_free_episodes'] for r in rows)/n for k in ['earlier_source_mse','later_source_mse']]
                lines.append('| '+name+' | '+' | '.join(map(number,vals))+' |')
        grid=model['grid']
        lines += ['', f"Grid native MSE: row={number(grid['native_mse']['row'])}, column={number(grid['native_mse']['column'])}.", '',
                  '| Route | Head | Key distance / spatial distance correlation | Key distance / temporal lag correlation |', '|---|---:|---:|---:|']
        for route, rows in grid['key_geometry'].items():
            for r in rows:
                lines.append(f"| {route} | {r['head']} | {number(r['spatial_distance_correlation'])} | {number(r['time_lag_correlation'])} |")
        lines += ['', '| Same-cell key swap | Pixel MSE | Correct-patch argmax by head (0–3) |', '|---|---:|---|']
        for name,r in grid['key_swaps'].items():
            lines.append(f"| {name} | {number(r['pixel_mse'])} | {', '.join(map(number,r['patch_accuracy_by_head']))} |")
        lines += ['', '| Route | Query lag (px) | Final-Q cosine by head (0–3) |', '|---|---:|---|']
        for route, lags in grid['query_lag_cosines'].items():
            for lag, vals in lags.items():
                lines.append(f"| {route} | {lag} | {', '.join(map(number,vals))} |")
        lines += ['', f"Measured suite time (excluding imports): {model['elapsed_s']:.1f}s.", '']
    lines += ['## Interpretation limits', ''] + ['- '+s for s in result['notes']]
    Path(destination).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    write_report(json.loads(args.results.read_text()), args.results.with_name('report.md'))
