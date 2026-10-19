#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import warnings

import common

warnings.filterwarnings('ignore')


def feature_groups(feats):
    g = {k: [] for k in ['flow_stats', 'flow_counts', 'unique_counts', 'port_stats',
                         'table_stats', 'ratios', 'rolling_10s', 'diffs']}
    for f in feats:
        if f.endswith('_diff'):
            g['diffs'].append(f)
        elif '10s' in f:
            g['rolling_10s'].append(f)
        elif f.endswith('_ratio') or f == 'flow_packets_per_flow':
            g['ratios'].append(f)
        elif f.startswith('port_'):
            g['port_stats'].append(f)
        elif f.startswith('table_'):
            g['table_stats'].append(f)
        elif f.startswith('no_of_unique_'):
            g['unique_counts'].append(f)
        elif f.startswith('no_of_'):
            g['flow_counts'].append(f)
        elif f.startswith('flow_'):
            g['flow_stats'].append(f)
        else:
            g.setdefault('other', []).append(f)
    return {k: v for k, v in g.items() if v}


def eval_subset(model, col_idx, X, y, groups, seeds):
    macs, ges, accs = [], [], []
    for seed in seeds:
        tr, te, _, _ = common.stratified_window_split(groups, y, test_frac=0.3, seed=seed)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[np.ix_(tr, col_idx)])
        Xte = sc.transform(X[np.ix_(te, col_idx)])
        est = common.make_estimator(model, 'multiclass', y[tr])
        est.fit(Xtr, y[tr])
        pred = est.predict(Xte)
        macs.append(f1_score(y[te], pred, average='macro', zero_division=0))
        f1p = f1_score(y[te], pred, average=None, labels=list(range(len(common.CLASS_NAMES))), zero_division=0)
        ges.append(f1p[4])
        accs.append(accuracy_score(y[te], pred))
    return np.mean(macs), np.std(macs), np.mean(ges), np.std(ges), np.mean(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='ablation_results')
    ap.add_argument('--models', nargs='*', default=common.MODEL_NAMES)
    ap.add_argument('--seeds', type=int, nargs='*', default=[42, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = common.load_dataset(args.data)
    feats = common.feature_columns(df)
    X = df[feats].values
    y = common.multiclass_labels(df)
    groups = common.window_groups(df, y)
    idx = {f: i for i, f in enumerate(feats)}
    grps = feature_groups(feats)
    configs = [('ALL', feats)] + [(k, v) for k, v in grps.items()]

    print("features=%d  groups=%s" % (len(feats), {k: len(v) for k, v in grps.items()}))
    rows = []
    for model in args.models:
        print("\n==== %s  (multiclass, window-grouped, %d seeds, untuned) ====" % (model, len(args.seeds)))
        print("  %-14s %4s   macro-F1        GoldenEye-F1" % ('group', 'nf'))
        for name, cols in configs:
            ci = [idx[c] for c in cols]
            mm, ms, gm, gs, acc = eval_subset(model, ci, X, y, groups, args.seeds)
            rows.append(dict(model=model, group=name, n_features=len(cols),
                             macro_f1_mean=mm, macro_f1_std=ms,
                             goldeneye_f1_mean=gm, goldeneye_f1_std=gs, accuracy_mean=acc))
            print("  %-14s %4d   %.3f +- %.3f   %.3f +- %.3f" % (name, len(cols), mm, ms, gm, gs))

    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'feature_group_ablation.csv'), index=False)
    print("\nSaved feature_group_ablation.csv to", args.out)


if __name__ == '__main__':
    main()
