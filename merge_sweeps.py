"""
Merges the _A and _B halves of a split mesh sweep back into the single
CSV that plot_mesh_convergence.py reads, and reports the change against
the pre-wind-fix values.
"""
import os
import pandas as pd

RESULTS = 'project/point_density_results/'

for glacier in ('gulkana', 'kennicott'):
    parts = []
    for tag in ('_A', '_B'):
        fn = os.path.join(RESULTS, f'{glacier}_mesh_convergence{tag}.csv')
        assert os.path.exists(fn), f'missing {fn}'
        parts.append(pd.read_csv(fn))

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset='point_spacing', keep='last')
    df = df.sort_values('actual_n_points').reset_index(drop=True)
    df = df.drop(columns=['mb_diff_from_median'], errors='ignore')
    df['mb_diff_from_median'] = df['mass_balance'] - df['mass_balance'].median()

    out = os.path.join(RESULTS, f'{glacier}_mesh_convergence.csv')
    df.to_csv(out, index=False)

    old = pd.read_csv(os.path.join(
        RESULTS, f'{glacier}_mesh_convergence_prewindfix.csv'))
    cmp = df.merge(old[['point_spacing', 'mass_balance']],
                   on='point_spacing', how='left', suffixes=('', '_old'))

    print('=' * 68)
    print(f'{glacier.upper()}  ({len(df)} resolutions)')
    print('=' * 68)
    print(f'{"h":>7} {"N":>7} {"before":>10} {"after":>10} {"moved":>9}')
    for _, r in cmp.iterrows():
        before = r['mass_balance_old']
        moved = r['mass_balance'] - before if pd.notna(before) else float('nan')
        bs = f'{before:>10.4f}' if pd.notna(before) else f'{"-":>10}'
        ms = f'{moved:>+9.4f}' if pd.notna(moved) else f'{"-":>9}'
        print(f'{r["point_spacing"]:>7.0f} {int(r["actual_n_points"]):>7} '
              f'{bs} {r["mass_balance"]:>10.4f} {ms}')

    fine = df.iloc[-1]['mass_balance']
    print()
    print(f'converged (finest) = {fine:.4f}')
    for label, frame, col in (('before', old, 'mass_balance'),
                              ('after', df, 'mass_balance')):
        f = frame.sort_values('actual_n_points')
        spread = f[col].max() - f[col].min()
        print(f'  {label:>6}: full-sweep spread = {spread:.4f}')
    print(f'  saved to {out}')
    print()
