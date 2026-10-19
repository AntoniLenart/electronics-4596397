#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OKABE = dict(black='#000000', orange='#E69F00', skyblue='#56B4E9', green='#009E73',
             yellow='#F0E442', blue='#0072B2', vermillion='#D55E00', purple='#CC79A7',
             gray='#999999')
CLASS_NAMES = ['Normal', 'SYN', 'UDP', 'ICMP', 'GoldenEye']
CLASS_COLORS = [OKABE['gray'], OKABE['orange'], OKABE['skyblue'], OKABE['green'], OKABE['vermillion']]
MODEL_NAMES = ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost']
INK, MUTED = '#1a1a1a', '#6b6b6b'


def _style(ax, title, ylabel, xlabel=''):
    ax.set_title(title, color=INK, fontsize=12, pad=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.grid(axis='y', color='#e6e6e6', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')
    ax.tick_params(colors=MUTED, labelsize=9)


def _barlabels(ax, bars, fmt='%.2f'):
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(fmt % h, (b.get_x() + b.get_width() / 2, h),
                    ha='center', va='bottom', fontsize=7.5, color=INK,
                    xytext=(0, 2), textcoords='offset points')


def plot_label_distribution(run_dir, out):
    oc = os.path.join(run_dir, 'dataset_v3-corrected.csv')
    oo = os.path.join(run_dir, 'dataset_v3.csv')
    if not (os.path.exists(oc) and os.path.exists(oo)):
        print("skip label_distribution (datasets not found)")
        return
    dc, do = pd.read_csv(oc), pd.read_csv(oo)

    def counts(d):
        return [int((d['attack'] == 0).sum()), int(d['attack_syn'].sum()), int(d['attack_udp'].sum()),
                int(d['attack_icmp'].sum()), int(d['attack_goldeneye'].sum())]
    c_corr, c_orig = counts(dc), counts(do)

    x = np.arange(len(CLASS_NAMES)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, c_orig, w, label='Snort labels (original)', color=OKABE['blue'], zorder=3)
    b2 = ax.bar(x + w/2, c_corr, w, label='Ground truth (corrected)', color=OKABE['vermillion'], zorder=3)
    _barlabels(ax, b1, '%d'); _barlabels(ax, b2, '%d')
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    _style(ax, '', 'Seconds (samples)')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'label_distribution.png'), dpi=160); plt.close(fig)
    print("wrote label_distribution.png")


def _load_grouped(windowgrouped_dir):
    """Return {(model,task): dataframe of per-fold rows}."""
    data = {}
    for m in MODEL_NAMES:
        for task in ('binary', 'multiclass'):
            p = os.path.join(windowgrouped_dir, '%s_%s_groupkfold.csv' % (m, task))
            if os.path.exists(p):
                data[(m, task)] = pd.read_csv(p)
    return data


def plot_leakage_contrast(windowgrouped_dir, random_dir, out):
    grouped = _load_grouped(windowgrouped_dir)
    rp = os.path.join(random_dir, 'random_split_summary.csv')
    if not grouped or not os.path.exists(rp):
        print("skip leakage_contrast (need windowgrouped + random results)")
        return
    rnd = pd.read_csv(rp)

    def rnd_val(model, task, col):
        row = rnd[(rnd['model'] == model) & (rnd['task'] == task)]
        return float(row[col].iloc[0]) if len(row) and col in row else np.nan

    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(MODEL_NAMES)); w = 0.38
    rg = [rnd_val(m, 'multiclass', 'f1_GoldenEye') for m in MODEL_NAMES]
    gg = [grouped[(m, 'multiclass')]['f1_GoldenEye'].mean() if (m, 'multiclass') in grouped else np.nan for m in MODEL_NAMES]
    gge = [grouped[(m, 'multiclass')]['f1_GoldenEye'].std() if (m, 'multiclass') in grouped else 0 for m in MODEL_NAMES]
    b1 = ax.bar(x - w/2, rg, w, label='Random split (leaky)', color=OKABE['gray'], zorder=3)
    b2 = ax.bar(x + w/2, gg, w, yerr=gge, capsize=3, label='Window-grouped', color=OKABE['vermillion'], zorder=3)
    _barlabels(ax, b1); _barlabels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=20, ha='right')
    ax.set_ylim(0, 1.05); _style(ax, '', 'GoldenEye F1'); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'leakage_contrast.png'), dpi=160); plt.close(fig)
    print("wrote leakage_contrast.png")


def plot_windowgrouped_perclass(windowgrouped_dir, out):
    grouped = _load_grouped(windowgrouped_dir)
    present = [m for m in MODEL_NAMES if (m, 'multiclass') in grouped]
    if not present:
        print("skip windowgrouped_perclass (no multiclass results)")
        return
    x = np.arange(len(CLASS_NAMES)); w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    model_colors = [OKABE['orange'], OKABE['skyblue'], OKABE['green'], OKABE['blue']]
    for i, m in enumerate(present):
        df = grouped[(m, 'multiclass')]
        means = [df['f1_%s' % c].mean() for c in CLASS_NAMES]
        stds = [df['f1_%s' % c].std() for c in CLASS_NAMES]
        ax.bar(x + (i - (len(present)-1)/2)*w, means, w, yerr=stds, capsize=2,
               label=m, color=model_colors[i % 4], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    _style(ax, '', 'F1')
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'windowgrouped_perclass_f1.png'), dpi=160); plt.close(fig)
    print("wrote windowgrouped_perclass_f1.png")


def plot_common_testset_regime(common_dir, out):
    p = os.path.join(common_dir, 'all_results_multiseed.csv')
    if not os.path.exists(p):
        print("skip common_testset_regime (no multiseed results)")
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    specs = [('multiclass', 'f1_macro', 'Macro-F1'),
             ('multiclass', 'f1_GoldenEye', 'GoldenEye F1')]
    x = np.arange(len(MODEL_NAMES)); w = 0.38
    for ax, (task, col, title) in zip(axes, specs):
        sub = df[df['task'] == task]
        def ms(model, regime):
            r = sub[(sub['model'] == model) & (sub['regime'] == regime)]
            return (r[col].mean(), r[col].std()) if len(r) and col in r else (np.nan, 0)
        om = [ms(m, 'original') for m in MODEL_NAMES]
        cm = [ms(m, 'corrected') for m in MODEL_NAMES]
        b1 = ax.bar(x - w/2, [a for a, _ in om], w, yerr=[b for _, b in om], capsize=3,
                    label='Trained on Snort labels', color=OKABE['blue'], zorder=3)
        b2 = ax.bar(x + w/2, [a for a, _ in cm], w, yerr=[b for _, b in cm], capsize=3,
                    label='Trained on ground truth', color=OKABE['vermillion'], zorder=3)
        _barlabels(ax, b1); _barlabels(ax, b2)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=20, ha='right')
        ax.set_ylim(0, 1.05); _style(ax, '', title); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'common_testset_regime.png'), dpi=160); plt.close(fig)
    print("wrote common_testset_regime.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=os.path.join('..', 'data'))
    ap.add_argument('--windowgrouped', default='windowgrouped_results')
    ap.add_argument('--random', default='random_split_results')
    ap.add_argument('--common', default='common_testset_results')
    ap.add_argument('--out', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plot_label_distribution(args.run_dir, args.out)
    plot_leakage_contrast(args.windowgrouped, args.random, args.out)
    plot_windowgrouped_perclass(args.windowgrouped, args.out)
    plot_common_testset_regime(args.common, args.out)
    print("\nFigures written to", args.out)


if __name__ == '__main__':
    main()
