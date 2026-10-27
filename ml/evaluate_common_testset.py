import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
import warnings

import common

warnings.filterwarnings('ignore')


def load_both(original_path, corrected_path):
    do = common.load_dataset(original_path)
    dc = common.load_dataset(corrected_path)
    feats = common.feature_columns(dc)
    assert list(do.columns) == list(dc.columns) or set(feats).issubset(do.columns), "column mismatch"
    if 'ts_epoch' in do.columns:
        assert np.allclose(do['ts_epoch'].values, dc['ts_epoch'].values), "timestamp mismatch"
    assert np.array_equal(do[feats].values, dc[feats].values), "feature values differ between regimes"
    return do, dc, feats


def run_one(model_name, task, X_tr, y_tr, X_te, y_te, groups_tr, out_dir, regime, seed):
    inner = GroupKFold(n_splits=min(3, len(np.unique(groups_tr))))
    search = common.make_search(model_name, task, y_tr, inner)
    search.fit(X_tr, y_tr, groups=groups_tr)
    pred = search.best_estimator_.predict(X_te)

    res = dict(model=model_name, task=task, regime=regime, seed=seed,
               best_params=str(search.best_params_))
    if task == 'binary':
        res.update(accuracy=accuracy_score(y_te, pred),
                   precision=precision_score(y_te, pred, zero_division=0),
                   recall=recall_score(y_te, pred, zero_division=0),
                   f1=f1_score(y_te, pred, zero_division=0))
    else:
        res.update(accuracy=accuracy_score(y_te, pred),
                   f1_macro=f1_score(y_te, pred, average='macro', zero_division=0))
        labels = list(range(len(common.CLASS_NAMES)))
        f1_per = f1_score(y_te, pred, average=None, labels=labels, zero_division=0)
        for c, name in enumerate(common.CLASS_NAMES):
            res['f1_%s' % name] = f1_per[c]
    np.save(os.path.join(out_dir, '%s_%s_%s_seed%d_pred.npy' % (model_name, task, regime, seed)), pred)
    return res, pred


def run_seed(seed, do, dc, feats, out_dir):
    X = dc[feats].values
    y_multi_c = common.multiclass_labels(dc)
    y_multi_o = common.multiclass_labels(do)
    y_bin_c, y_bin_o = dc['attack'].values, do['attack'].values
    groups = common.window_groups(dc, y_multi_c)

    tr, te, _, _ = common.stratified_window_split(groups, y_multi_c, test_frac=0.3, seed=seed)
    print("\n### seed %d: train rows=%d test rows=%d | train windows=%d test windows=%d" % (
        seed, len(tr), len(te), len(np.unique(groups[tr])), len(np.unique(groups[te]))))

    X_tr = X[tr]
    X_te = X[te]
    g_tr = groups[tr]

    results, mcnemars = [], []
    for task in ('binary', 'multiclass'):
        if task == 'binary':
            yo, yc, ygt = y_bin_o[tr], y_bin_c[tr], y_bin_c[te]
        else:
            yo, yc, ygt = y_multi_o[tr], y_multi_c[tr], y_multi_c[te]
        for model_name in common.MODEL_NAMES:
            ro, po = run_one(model_name, task, X_tr, yo, X_te, ygt, g_tr, out_dir, 'original', seed)
            rc, pc = run_one(model_name, task, X_tr, yc, X_te, ygt, g_tr, out_dir, 'corrected', seed)
            results += [ro, rc]
            mc = common.mcnemar_test(ygt, po, pc)
            mc.update(seed=seed, model=model_name, task=task)
            mcnemars.append(mc)
            key = 'f1' if task == 'binary' else 'f1_macro'
            print("  %-18s %-10s orig %s=%.3f | corr %s=%.3f | McNemar p=%.3f" % (
                model_name, task, key, ro[key], key, rc[key], mc['p_value']))
    return results, mcnemars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original', default=os.path.join('..', 'data', 'dataset_v3.csv'))
    ap.add_argument('--corrected', default=os.path.join('..', 'data', 'dataset_v3-corrected.csv'))
    ap.add_argument('--out', default='common_testset_results')
    ap.add_argument('--seeds', type=int, nargs='*', default=[42, 1, 2, 3, 4])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    do, dc, feats = load_both(args.original, args.corrected)
    all_results, all_mcnemar = [], []
    for seed in args.seeds:
        r, m = run_seed(seed, do, dc, feats, args.out)
        all_results += r
        all_mcnemar += m

    res = pd.DataFrame(all_results)
    res.to_csv(os.path.join(args.out, 'all_results_multiseed.csv'), index=False)
    metric_cols = [c for c in ['accuracy', 'precision', 'recall', 'f1', 'f1_macro'] +
                   ['f1_%s' % n for n in common.CLASS_NAMES] if c in res.columns]
    summary = res.groupby(['model', 'task', 'regime'])[metric_cols].agg(['mean', 'std'])
    summary.to_csv(os.path.join(args.out, 'summary_multiseed.csv'))
    print("\n=== SUMMARY (mean +- std across %d seeds) ===" % len(args.seeds))
    with pd.option_context('display.max_rows', None, 'display.width', 250):
        print(summary)

    mc = pd.DataFrame(all_mcnemar)
    mc.to_csv(os.path.join(args.out, 'mcnemar_multiseed.csv'), index=False)
    print("\n=== McNemar (significant seeds, alpha=0.05) ===")
    for (model, task), g in mc.groupby(['model', 'task']):
        sig = g[g['p_value'] < 0.05]
        if len(sig):
            direction = "corrected-better in %d, original-better in %d" % (
                int((sig['n10'] > sig['n01']).sum()), int((sig['n01'] > sig['n10']).sum()))
        else:
            direction = "n/a"
        print("  %-18s %-10s: %d/%d significant (%s); median p=%.4f" % (
            model, task, len(sig), len(args.seeds), direction, g['p_value'].median()))


if __name__ == '__main__':
    main()
