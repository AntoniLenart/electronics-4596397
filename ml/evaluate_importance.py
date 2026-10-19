#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

import common

warnings.filterwarnings('ignore')


def tune_fit(model_name, task, Xtr, ytr, gtr):
    cfg = common.MODEL_GRIDS[model_name][task]
    est = common.make_estimator(model_name, task, ytr)
    inner = GroupKFold(n_splits=min(3, len(np.unique(gtr))))
    s = GridSearchCV(est, cfg['grid'], cv=inner, scoring=cfg['scoring'], n_jobs=-1, refit=True)
    s.fit(Xtr, ytr, groups=gtr)
    return s.best_estimator_


def scorer_for(task):
    if task == 'binary':
        return lambda yt, yp: f1_score(yt, yp, zero_division=0)
    return lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)


def build_clusters(Xdf, feats, thr):
    """Hierarchically cluster features by |Spearman|; cut so within-cluster |r|>=thr.
    Returns (abs_corr, clusters{cid:[col_idx...]}, leaf_order)."""
    corr = Xdf[feats].corr(method='spearman').abs().fillna(0).values
    d = 1.0 - corr
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    Z = linkage(squareform(d, checks=False), method='average')
    labels = fcluster(Z, t=1.0 - thr, criterion='distance')
    clusters = {}
    for i, l in enumerate(labels):
        clusters.setdefault(int(l), []).append(i)
    return corr, clusters, leaves_list(Z)


def cluster_label(cols, feats, corr):
    if len(cols) == 1:
        return feats[cols[0]]
    sub = corr[np.ix_(cols, cols)]
    medoid = cols[int(np.argmax(sub.sum(axis=1)))]
    return "%s (+%d)" % (feats[medoid], len(cols) - 1)


def cluster_perm_importance(est, Xte, yte, clusters, scorer, n_repeats, rng):
    """Jointly permute each cluster's columns on the fitted model; mean score drop."""
    base = scorer(yte, est.predict(Xte))
    out = {}
    for cid, cols in clusters.items():
        cols = np.asarray(cols)
        drops = []
        for _ in range(n_repeats):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xte))
            Xp[:, cols] = Xte[np.ix_(perm, cols)]
            drops.append(base - scorer(yte, est.predict(Xp)))
        out[cid] = (float(np.mean(drops)), float(np.std(drops)))
    return base, out


def correlation_heatmap(corr, order, out):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr[np.ix_(order, order)], cmap='magma', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label='|Spearman r|')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'correlation_heatmap.png'), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='importance_results')
    ap.add_argument('--models', nargs='*', default=common.MODEL_NAMES)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-repeats', type=int, default=10)
    ap.add_argument('--corr-threshold', type=float, default=0.8,
                    help='|Spearman| at/above which features are treated as one cluster (default 0.8)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = common.load_dataset(args.data)
    feats = common.feature_columns(df)
    X = df[feats].values
    y_multi = common.multiclass_labels(df)
    y_bin = df['attack'].values
    groups = common.window_groups(df, y_multi)

    tr, te, _, _ = common.stratified_window_split(groups, y_multi, test_frac=0.3, seed=args.seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te]); gtr = groups[tr]
    print("train rows=%d test rows=%d | train windows=%d test windows=%d | features=%d" % (
        len(tr), len(te), len(np.unique(groups[tr])), len(np.unique(groups[te])), len(feats)))

    corr, clusters, order = build_clusters(df, feats, args.corr_threshold)
    print("\nFeature correlation: %d features -> %d independent clusters at |r|>=%.2f" % (
        len(feats), len(clusters), args.corr_threshold))
    clabels = {cid: cluster_label(cols, feats, corr) for cid, cols in clusters.items()}
    pd.DataFrame([dict(cluster=cid, size=len(cols), label=clabels[cid],
                       features='|'.join(feats[i] for i in cols))
                 for cid, cols in clusters.items()]).to_csv(
        os.path.join(args.out, 'feature_clusters.csv'), index=False)
    correlation_heatmap(corr, order, args.out)

    rng = np.random.RandomState(args.seed)
    per_feat_rows, cluster_rows = [], []
    for model in args.models:
        for task, (ytr, yte, scoring) in [('binary', (y_bin[tr], y_bin[te], 'f1')),
                                          ('multiclass', (y_multi[tr], y_multi[te], 'f1_macro'))]:
            est = tune_fit(model, task, Xtr, ytr, gtr)

            r = permutation_importance(est, Xte, yte, scoring=scoring,
                                       n_repeats=args.n_repeats, random_state=args.seed, n_jobs=-1)
            for i, f in enumerate(feats):
                per_feat_rows.append(dict(model=model, task=task, feature=f,
                                          importance=float(r.importances_mean[i]), std=float(r.importances_std[i])))

            base, cimp = cluster_perm_importance(est, Xte, yte, clusters, scorer_for(task), args.n_repeats, rng)
            for cid, (mu, sd) in cimp.items():
                cluster_rows.append(dict(model=model, task=task, cluster=cid, label=clabels[cid],
                                         size=len(clusters[cid]), importance=mu, std=sd))
            top = sorted(cimp.items(), key=lambda kv: -kv[1][0])[:8]
            print("\n[%s %s] top-8 CLUSTER importance (joint-permute %s drop; base=%.3f):" % (
                model, task, scoring, base))
            for cid, (mu, sd) in top:
                print("   %-42s %.4f +- %.4f" % (clabels[cid][:42], mu, sd))

    pd.DataFrame(per_feat_rows).to_csv(os.path.join(args.out, 'permutation_importance.csv'), index=False)
    pd.DataFrame(cluster_rows).to_csv(os.path.join(args.out, 'cluster_importance.csv'), index=False)

    pf = pd.DataFrame(per_feat_rows)
    cf = pd.DataFrame(cluster_rows)
    fig_model = 'XGBoost' if 'XGBoost' in args.models else args.models[0]
    for src, col, fname, title in [
        (pf, 'feature', 'permutation_importance_%s_multiclass.png' % fig_model,
         '%s multiclass - per-feature permutation importance (top 15)' % fig_model),
        (cf, 'label', 'cluster_importance_%s_multiclass.png' % fig_model,
         '%s multiclass - correlation-cluster importance (top 15, multicollinearity-robust)' % fig_model)]:
        sub = src[(src['model'] == fig_model) & (src['task'] == 'multiclass')].nlargest(15, 'importance')[::-1]
        if len(sub):
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.barh(sub[col].astype(str), sub['importance'], xerr=sub['std'], color='#0072B2', zorder=3)
            ax.set_title(title, fontsize=10.5)
            ax.set_xlabel('mean macro-F1 decrease when permuted')
            ax.grid(axis='x', color='#e6e6e6', zorder=0); ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            fig.tight_layout(); fig.savefig(os.path.join(args.out, fname), dpi=160); plt.close(fig)

    print("\nSaved permutation_importance.csv, cluster_importance.csv, feature_clusters.csv,"
          " correlation_heatmap.png to", args.out)


if __name__ == '__main__':
    main()
