import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
import warnings

import common

warnings.filterwarnings('ignore')


def evaluate(model_name, task, X, y, out_dir, seed=42):
    cfg = common.MODEL_GRIDS[model_name][task]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    est = common.make_estimator(model_name, task, y_tr)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(est, cfg['grid'], cv=cv, scoring=cfg['scoring'], n_jobs=-1, refit=True)
    search.fit(X_tr_s, y_tr)
    pred = search.best_estimator_.predict(X_te_s)

    if task == 'binary':
        row = dict(model=model_name, task=task,
                   accuracy=accuracy_score(y_te, pred),
                   precision=precision_score(y_te, pred, zero_division=0),
                   recall=recall_score(y_te, pred, zero_division=0),
                   f1=f1_score(y_te, pred, zero_division=0),
                   best_params=json.dumps(search.best_params_))
    else:
        f1_per = f1_score(y_te, pred, average=None, labels=list(range(len(common.CLASS_NAMES))), zero_division=0)
        row = dict(model=model_name, task=task,
                   accuracy=accuracy_score(y_te, pred),
                   f1_macro=f1_score(y_te, pred, average='macro', zero_division=0),
                   best_params=json.dumps(search.best_params_))
        for c, name in enumerate(common.CLASS_NAMES):
            row['f1_%s' % name] = f1_per[c]
    print("  [%s %s] acc=%.4f %s" % (
        model_name, task, row['accuracy'],
        ('f1=%.4f' % row['f1']) if task == 'binary' else ('f1_macro=%.4f' % row['f1_macro'])))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join('..', 'data', 'dataset_v3-corrected.csv'))
    ap.add_argument('--out', default='random_split_results')
    ap.add_argument('--models', nargs='*', default=common.MODEL_NAMES)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = common.load_dataset(args.data)
    feats = common.feature_columns(df)
    X = df[feats].values
    y_multi = common.multiclass_labels(df)
    y_bin = df['attack'].values

    rows = []
    for model_name in args.models:
        print("\n%s (random 70/30 split - LEAKAGE DEMO)" % model_name)
        rows.append(evaluate(model_name, 'binary', X, y_bin, args.out, args.seed))
        rows.append(evaluate(model_name, 'multiclass', X, y_multi, args.out, args.seed))

    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'random_split_summary.csv'), index=False)
    print("\nSaved random_split_summary.csv (compare against windowgrouped_results/summary_all_models.csv)")


if __name__ == '__main__':
    main()
