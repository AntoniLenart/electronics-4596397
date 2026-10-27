import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
import warnings

import common

warnings.filterwarnings('ignore')


def nested_eval(model_name, task, X, y, groups, out_dir, n_outer=5, n_inner=3):
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y, groups=groups)):
        X_tr, X_te, y_tr, y_te, g_tr = X[tr], X[te], y[tr], y[te], groups[tr]

        inner = GroupKFold(n_splits=min(n_inner, len(np.unique(g_tr))))
        search = common.make_search(model_name, task, y_tr, inner)
        search.fit(X_tr, y_tr, groups=g_tr)
        pred = search.best_estimator_.predict(X_te)

        if task == 'binary':
            row = dict(fold=fold, n_train=len(tr), n_test=len(te),
                       test_attack_frac=float(np.mean(y_te)),
                       accuracy=accuracy_score(y_te, pred),
                       precision=precision_score(y_te, pred, zero_division=0),
                       recall=recall_score(y_te, pred, zero_division=0),
                       f1=f1_score(y_te, pred, zero_division=0),
                       best_params=json.dumps(search.best_params_))
        else:
            f1_per = f1_score(y_te, pred, average=None, labels=list(range(len(common.CLASS_NAMES))), zero_division=0)
            row = dict(fold=fold, n_train=len(tr), n_test=len(te),
                       accuracy=accuracy_score(y_te, pred),
                       f1_macro=f1_score(y_te, pred, average='macro', zero_division=0),
                       best_params=json.dumps(search.best_params_))
            for c, name in enumerate(common.CLASS_NAMES):
                row['f1_%s' % name] = f1_per[c]
                row['n_test_%s' % name] = int(np.sum(y_te == c))
        rows.append(row)
        print("  [%s %s] fold %d: acc=%.4f %s" % (
            model_name, task, fold, row['accuracy'],
            ('f1=%.4f' % row['f1']) if task == 'binary' else ('f1_macro=%.4f' % row['f1_macro'])))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, '%s_%s_groupkfold.csv' % (model_name, task)), index=False)
    metrics = ['accuracy', 'precision', 'recall', 'f1'] if task == 'binary' else \
              ['accuracy', 'f1_macro'] + ['f1_%s' % n for n in common.CLASS_NAMES]
    summary = {m: (df[m].mean(), df[m].std()) for m in metrics if m in df}
    for m, (mu, sd) in summary.items():
        print("    %-12s %.4f +- %.4f" % (m, mu, sd))
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('..', 'data', 'dataset_v3-corrected.csv'))
    ap.add_argument('--out', default='windowgrouped_results')
    ap.add_argument('--models', nargs='*', default=common.MODEL_NAMES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = common.load_dataset(args.data)
    feats = common.feature_columns(df)
    X = df[feats].values
    y_multi = common.multiclass_labels(df)
    y_bin = df['attack'].values
    groups = common.window_groups(df, y_multi)
    print("Rows: %d | windows: %d | features: %d" % (len(df), len(np.unique(groups)), len(feats)))
    for c, name in enumerate(common.CLASS_NAMES):
        print("  %-9s windows=%d rows=%d" % (
            name, len(np.unique(groups[y_multi == c])), int(np.sum(y_multi == c))))

    all_summaries = []
    for model_name in args.models:
        print("\n" + "=" * 70 + "\n%s - BINARY (nested GroupKFold)\n" % model_name + "=" * 70)
        _, sb = nested_eval(model_name, 'binary', X, y_bin, groups, args.out)
        print("\n" + "=" * 70 + "\n%s - MULTICLASS (nested GroupKFold)\n" % model_name + "=" * 70)
        _, sm = nested_eval(model_name, 'multiclass', X, y_multi, groups, args.out)
        for task, s in (('binary', sb), ('multiclass', sm)):
            for m, (mu, sd) in s.items():
                all_summaries.append(dict(model=model_name, task=task, metric=m, mean=mu, std=sd))
    pd.DataFrame(all_summaries).to_csv(os.path.join(args.out, 'summary_all_models.csv'), index=False)
    print("\nSaved summary_all_models.csv to", args.out)


if __name__ == '__main__':
    main()
