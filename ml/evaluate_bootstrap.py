import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import common

GE = common.CLASS_NAMES.index('GoldenEye')
LABELS = list(range(len(common.CLASS_NAMES)))


def macro_and_ge(y, p):
    per = f1_score(y, p, average=None, labels=LABELS, zero_division=0)
    return float(per.mean()), float(per[GE])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corrected', required=True)
    ap.add_argument('--preds', required=True, help='dir with *_pred.npy from evaluate_common_testset')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=[42, 1, 2, 3, 4])
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--task', default='multiclass')
    args = ap.parse_args()
    out = args.out or args.preds
    os.makedirs(out, exist_ok=True)

    dc = common.load_dataset(args.corrected)
    y_multi = common.multiclass_labels(dc)
    y_bin = dc['attack'].values
    groups = common.window_groups(dc, y_multi)

    rng = np.random.RandomState(12345)
    rows = []
    for seed in args.seeds:
        tr, te, _, _ = common.stratified_window_split(groups, y_multi, test_frac=0.3, seed=seed)
        g_te = groups[te]
        y_te = y_multi[te] if args.task == 'multiclass' else y_bin[te]
        uniq = np.unique(g_te)
        wpos = {w: np.where(g_te == w)[0] for w in uniq}

        for model in common.MODEL_NAMES:
            fo = os.path.join(args.preds, '%s_%s_original_seed%d_pred.npy' % (model, args.task, seed))
            fc = os.path.join(args.preds, '%s_%s_corrected_seed%d_pred.npy' % (model, args.task, seed))
            if not (os.path.exists(fo) and os.path.exists(fc)):
                print("skip %s seed %d (preds missing)" % (model, seed))
                continue
            po, pc = np.load(fo), np.load(fc)
            assert len(po) == len(pc) == len(y_te), \
                "length mismatch %s seed %d: po=%d pc=%d te=%d" % (model, seed, len(po), len(pc), len(y_te))

            mc_c, ge_c = macro_and_ge(y_te, pc)
            mc_o, ge_o = macro_and_ge(y_te, po)
            obs_macro, obs_ge = mc_c - mc_o, ge_c - ge_o

            dmac = np.empty(args.nboot)
            dge = np.empty(args.nboot)
            for b in range(args.nboot):
                samp = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([wpos[w] for w in samp])
                yy = y_te[idx]
                a_mac, a_ge = macro_and_ge(yy, pc[idx])
                b_mac, b_ge = macro_and_ge(yy, po[idx])
                dmac[b] = a_mac - b_mac
                dge[b] = a_ge - b_ge

            lmac, umac = np.percentile(dmac, [2.5, 97.5])
            lge, uge = np.percentile(dge, [2.5, 97.5])
            rows.append(dict(
                seed=seed, model=model, task=args.task,
                n_test_windows=int(len(uniq)), n_test_rows=int(len(te)),
                delta_macro_f1=obs_macro, macro_ci_lo=lmac, macro_ci_hi=umac,
                macro_frac_gt0=float(np.mean(dmac > 0)),
                delta_goldeneye_f1=obs_ge, ge_ci_lo=lge, ge_ci_hi=uge,
                ge_frac_gt0=float(np.mean(dge > 0))))
            print("seed %2d %-18s dMacro=%+.3f [%+.3f,%+.3f] P(>0)=%.3f | dGE=%+.3f [%+.3f,%+.3f] P(>0)=%.3f" % (
                seed, model, obs_macro, lmac, umac, float(np.mean(dmac > 0)),
                obs_ge, lge, uge, float(np.mean(dge > 0))))

    df = pd.DataFrame(rows)
    path = os.path.join(out, 'bootstrap_%s.csv' % args.task)
    df.to_csv(path, index=False)
    print("\n=== across-seed summary (observed delta, mean [min,max]; frac of seeds with CI fully >0) ===")
    for model, g in df.groupby('model'):
        macro_pos = int(((g['macro_ci_lo'] > 0)).sum())
        ge_pos = int(((g['ge_ci_lo'] > 0)).sum())
        print("  %-18s dMacro mean=%+.3f [%+.3f,%+.3f] (CI>0 in %d/%d) | dGE mean=%+.3f [%+.3f,%+.3f] (CI>0 in %d/%d)" % (
            model,
            g['delta_macro_f1'].mean(), g['delta_macro_f1'].min(), g['delta_macro_f1'].max(), macro_pos, len(g),
            g['delta_goldeneye_f1'].mean(), g['delta_goldeneye_f1'].min(), g['delta_goldeneye_f1'].max(), ge_pos, len(g)))
    print("\nwrote", path)


if __name__ == '__main__':
    main()
