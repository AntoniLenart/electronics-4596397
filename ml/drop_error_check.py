import os
import argparse
import numpy as np
import pandas as pd

LOCAL_PORT = 4294967294
COUNTERS = ['rx_dropped', 'tx_dropped', 'rx_errors', 'tx_errors']
# retained rows plus the dropped cooldown rows give the window type of every analysed second
SECOND_FILES = ['dataset_v3-corrected.csv', 'cooldown_rows.csv']


def window_type_by_second(run_dir):
    sec = pd.concat([pd.read_csv(os.path.join(run_dir, f), usecols=['ts_epoch', 'window_type'])
                     for f in SECOND_FILES])
    sec = sec.drop_duplicates('ts_epoch')
    return pd.Series(sec['window_type'].values, index=np.floor(sec['ts_epoch']).astype(np.int64))


def check_run(run_dir):
    ports = pd.read_csv(os.path.join(run_dir, 'port_stats.csv'),
                        usecols=['ts_epoch', 'switch_dpid', 'port_no'] + COUNTERS)
    wtype = window_type_by_second(run_dir)

    data = ports[ports.port_no != LOCAL_PORT]
    local = ports[ports.port_no == LOCAL_PORT].sort_values('ts_epoch')
    print("\n%s" % run_dir)
    print("  data ports: max of %s = %d" % ('/'.join(COUNTERS), int(data[COUNTERS].max().max())))
    print("  LOCAL port: max of tx_dropped/rx_errors/tx_errors = %d" %
          int(local[['tx_dropped', 'rx_errors', 'tx_errors']].max().max()))

    stats = {}
    for _, g in local.groupby('switch_dpid'):
        inc = np.diff(g['rx_dropped'].values)
        types = pd.Series(np.floor(g['ts_epoch'].values[1:]).astype(np.int64)).map(wtype)
        for t, x in zip(types, inc):
            if pd.isna(t):
                continue
            s = stats.setdefault(t, [0, 0, 0.0])
            s[0] += 1
            if x > 0:
                s[1] += 1
                s[2] += float(x)
    print("  LOCAL rx_dropped increments per poll, by window type:")
    for t in sorted(stats):
        n, pos, total = stats[t]
        print("    %-10s %5.1f%% of %5d polls increment, mean %.2f packets when they do" % (
            t, 100.0 * pos / n, n, total / pos if pos else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, nargs='+',
                    help='telemetry run dirs (port_stats.csv, dataset_v3-corrected.csv, cooldown_rows.csv)')
    args = ap.parse_args()
    for run in args.runs:
        check_run(run)


if __name__ == '__main__':
    main()
