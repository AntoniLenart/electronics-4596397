import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import common


def score(y, p, task):
    if task == 'binary':
        return dict(accuracy=accuracy_score(y, p),
                    precision=precision_score(y, p, zero_division=0),
                    recall=recall_score(y, p, zero_division=0),
                    f1=f1_score(y, p, zero_division=0))
    per = f1_score(y, p, average=None, labels=list(range(len(common.CLASS_NAMES))), zero_division=0)
    d = dict(accuracy=accuracy_score(y, p), f1_macro=f1_score(y, p, average='macro', zero_division=0))
    for c, n in enumerate(common.CLASS_NAMES):
        d['f1_' + n] = per[c]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original', required=True, help='combined Snort-label dataset (dataset_v3.csv, with campaign column)')
    ap.add_argument('--corrected', required=True, help='combined ground-truth dataset (same rows/features)')
    ap.add_argument('--held-campaign', type=int, default=3, help='campaign index held out as the test set (3 = C4)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='*', default=common.MODEL_NAMES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df_o = common.load_dataset(args.original)
    df_c = common.load_dataset(args.corrected)
    feats = common.feature_columns(df_c)
    X = df_c[feats].values
    assert np.array_equal(df_o[feats].values, X), "regime feature values differ (should be identical)"
    campaigns = df_c['campaign'].values
    assert np.array_equal(df_o['campaign'].values, campaigns), "regime campaign columns differ"
    tr, te = campaigns != args.held_campaign, campaigns == args.held_campaign
    wid = common.window_groups(df_c, common.multiclass_labels(df_c))
    print("train rows %d (campaigns %s) | test rows %d (campaign %d)" % (
        tr.sum(), sorted(np.unique(campaigns[tr])), te.sum(), args.held_campaign))

    rows = []
    for task in ('binary', 'multiclass'):
        if task == 'binary':
            y_o, y_c = df_o['attack'].values, df_c['attack'].values
        else:
            y_o, y_c = common.multiclass_labels(df_o), common.multiclass_labels(df_c)
        y_test = y_c[te]
        n_train_windows = len(np.unique(wid[tr]))
        inner = GroupKFold(n_splits=min(3, n_train_windows))
        for model in args.models:
            for regime, y in (('ids', y_o), ('groundtruth', y_c)):
                ytrain = y[tr]
                s = common.make_search(model, task, ytrain, inner)
                s.fit(X[tr], ytrain, groups=wid[tr])
                p = s.best_estimator_.predict(X[te])
                r = dict(model=model, task=task, train_labels=regime, best_params=json.dumps(s.best_params_))
                r.update(score(y_test, p, task))
                rows.append(r)
                key = 'f1' if task == 'binary' else 'f1_macro'
                extra = '' if task == 'binary' else ' GE=%.3f' % r['f1_GoldenEye']
                print("  %-18s %-10s train=%-11s %s=%.3f%s" % (model, task, regime, key, r[key], extra))

    df = pd.DataFrame(rows)
    path = os.path.join(args.out, 'ids_to_c4.csv')
    df.to_csv(path, index=False)
    print("\nwrote", path)


if __name__ == '__main__':
    main()
