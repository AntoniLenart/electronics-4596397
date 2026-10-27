import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

import common

warnings.filterwarnings('ignore')


def run(model_name, task, X, y, campaigns, wid, out_dir):
    logo = LeaveOneGroupOut()
    rows = []
    for held, (tr, te) in zip(sorted(np.unique(campaigns)), logo.split(X, y, groups=campaigns)):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        n_train_windows = len(np.unique(wid[tr]))
        inner = GroupKFold(n_splits=min(3, n_train_windows))
        search = common.make_search(model_name, task, y_tr, inner)
        search.fit(X_tr, y_tr, groups=wid[tr])
        pred = search.best_estimator_.predict(X_te)

        if task == 'binary':
            row = dict(held_campaign=int(held), n_test=len(te),
                       accuracy=accuracy_score(y_te, pred),
                       precision=precision_score(y_te, pred, zero_division=0),
                       recall=recall_score(y_te, pred, zero_division=0),
                       f1=f1_score(y_te, pred, zero_division=0),
                       best_params=json.dumps(search.best_params_))
        else:
            f1_per = f1_score(y_te, pred, average=None, labels=list(range(len(common.CLASS_NAMES))), zero_division=0)
            row = dict(held_campaign=int(held), n_test=len(te),
                       accuracy=accuracy_score(y_te, pred),
                       f1_macro=f1_score(y_te, pred, average='macro', zero_division=0),
                       best_params=json.dumps(search.best_params_))
            for c, name in enumerate(common.CLASS_NAMES):
                row['f1_%s' % name] = f1_per[c]
        rows.append(row)
        print("  [%s %s] held-out campaign %s: acc=%.4f %s" % (
            model_name, task, held, row['accuracy'],
            ('f1=%.4f' % row['f1']) if task == 'binary' else ('f1_macro=%.4f' % row['f1_macro'])))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, '%s_%s_cross_campaign.csv' % (model_name, task)), index=False)
    metrics = ['accuracy', 'precision', 'recall', 'f1'] if task == 'binary' else \
              ['accuracy', 'f1_macro'] + ['f1_%s' % n for n in common.CLASS_NAMES]
    for m in metrics:
        if m in df:
            print("    %-12s %.4f +- %.4f" % (m, df[m].mean(), df[m].std()))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='combined corrected dataset (with a campaign column)')
    ap.add_argument('--out', default='cross_campaign_results')
    ap.add_argument('--models', nargs='*', default=common.MODEL_NAMES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = common.load_dataset(args.data)
    if 'campaign' not in df.columns or df['campaign'].nunique() < 2:
        raise SystemExit("Need a combined dataset with >=2 campaigns (see combine_campaigns.py)")
    feats = common.feature_columns(df)
    X = df[feats].values
    y_bin = df['attack'].values
    y_multi = common.multiclass_labels(df)
    campaigns = df['campaign'].values
    wid = common.window_groups(df, y_multi)
    print("Campaigns: %d | rows: %d | windows: %d | features: %d" % (
        df['campaign'].nunique(), len(df), len(np.unique(wid)), len(feats)))

    for model_name in args.models:
        print("\n" + "=" * 70 + "\n%s - BINARY (leave-one-campaign-out)\n" % model_name + "=" * 70)
        run(model_name, 'binary', X, y_bin, campaigns, wid, args.out)
        print("\n" + "=" * 70 + "\n%s - MULTICLASS (leave-one-campaign-out)\n" % model_name + "=" * 70)
        run(model_name, 'multiclass', X, y_multi, campaigns, wid, args.out)
    print("\nSaved cross-campaign results to", args.out)


if __name__ == '__main__':
    main()
