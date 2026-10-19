#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

WID_STRIDE = 10_000_000


def _load_tagged(run_dir, name, campaign):
    df = pd.read_csv(os.path.join(run_dir, name))
    if 'window_id' not in df.columns:
        raise ValueError("%s/%s has no window_id (regenerate with the v3 generate_dataset)" % (run_dir, name))
    df.insert(0, 'campaign', campaign)
    df['window_id'] = df['window_id'].astype('int64') + campaign * WID_STRIDE
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='campaign run dirs (each with dataset_v3*.csv)')
    ap.add_argument('--out', required=True, help='output dir for the combined dataset')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name in ('dataset_v3-corrected.csv', 'dataset_v3.csv'):
        parts, ref_cols = [], None
        print("\n=== %s ===" % name)
        for i, rd in enumerate(args.runs):
            df = _load_tagged(rd, name, i)
            feat_cols = [c for c in df.columns if c not in ('campaign',)]
            if ref_cols is None:
                ref_cols = feat_cols
            elif feat_cols != ref_cols:
                raise ValueError("column mismatch between campaigns (%s vs campaign 0)" % rd)
            n_attack = int(df['attack'].sum()) if 'attack' in df else -1
            print("  campaign %d  %-45s rows=%d windows=%d attack=%d"
                  % (i, os.path.basename(rd.rstrip('/\\')), len(df), df['window_id'].nunique(), n_attack))
            parts.append(df)
        comb = pd.concat(parts, ignore_index=True)
        assert comb.groupby('window_id')['campaign'].nunique().max() == 1, "window_id collided across campaigns"
        out = os.path.join(args.out, name)
        comb.to_csv(out, index=False)
        print("  -> combined rows=%d  windows=%d  campaigns=%d  attack=%d  -> %s"
              % (len(comb), comb['window_id'].nunique(), comb['campaign'].nunique(),
                 int(comb['attack'].sum()) if 'attack' in comb else -1, out))

    print("\nCombined dataset written to %s" % args.out)
    print("Train on it with the usual scripts (window-grouped split groups by the now-global window_id),")
    print("or use the `campaign` column for leave-one-campaign-out (evaluate_cross_campaign.py).")


if __name__ == '__main__':
    main()
