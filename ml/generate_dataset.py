#!/usr/bin/env python3
import os
import re
import sys
import argparse
import calendar
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ATTACK_LABELS = ['syn', 'udp', 'icmp', 'goldeneye']


WINDOW_RE = re.compile(
    r'WINDOW_(START|END)\s+id=(\d+)\s+kind=(\w+)\s+type=(\w+)\s+epoch=([\d.]+)')


def parse_sim_log(path):
    """Return (run_start_epoch, run_end_epoch, windows) where windows is a list of
    dicts {id, kind, type, start_epoch, end_epoch} sorted by start."""
    opens = {}
    windows = []
    run_start = run_end = None
    with open(path, 'r') as f:
        for line in f:
            m = WINDOW_RE.search(line)
            if m:
                ev, wid, kind, wtype, epoch = m.group(1), int(m.group(2)), m.group(3), m.group(4), float(m.group(5))
                if ev == 'START':
                    opens[wid] = (kind, wtype, epoch)
                else:
                    if wid in opens:
                        kind0, wtype0, start_epoch = opens.pop(wid)
                        windows.append(dict(id=wid, kind=kind0, type=wtype0,
                                            start_epoch=start_epoch, end_epoch=epoch))
                continue
            if 'STARTED' in line or 'ENDED' in line:
                pass
    if not windows:
        raise ValueError("No WINDOW_START/END markers found in %s (is this a v3 sim log?)" % path)
    windows.sort(key=lambda w: w['start_epoch'])
    run_start = windows[0]['start_epoch']
    run_end = windows[-1]['end_epoch']
    return run_start, run_end, windows


def parse_alerts(path, run_start_epoch, run_end_epoch):
    """Parse Snort alerts (CSV or fast format) into (epoch, attack_type) pairs.

    Timestamps are MM/DD-HH:MM:SS.ffffff with no year; Snort is run with -U so
    they are UTC. We anchor the year to the run start (UTC) and roll over a year
    boundary if needed, then convert as UTC via calendar.timegm.
    """
    anchor_year = datetime.fromtimestamp(run_start_epoch, tz=timezone.utc).year
    ts_re = re.compile(r'(\d{2})/(\d{2})-(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?')
    out = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            low = line.lower()
            if 'syn flood' in low:
                atype = 'syn'
            elif 'udp flood' in low:
                atype = 'udp'
            elif 'icmp flood' in low or 'many pings' in low:
                atype = 'icmp'
            elif 'goldeneye' in low or 'keepalive' in low or 'keep-alive' in low:
                atype = 'goldeneye'
            else:
                continue
            m = ts_re.search(line)
            if not m:
                continue
            mon, day, hh, mm, ss = (int(m.group(i)) for i in range(1, 6))
            frac = float('0.' + m.group(6)) if m.group(6) else 0.0
            epoch = calendar.timegm((anchor_year, mon, day, hh, mm, ss, 0, 0, 0)) + frac
            if epoch < run_start_epoch - 43200:
                epoch = calendar.timegm((anchor_year + 1, mon, day, hh, mm, ss, 0, 0, 0)) + frac
            out.append((epoch, atype))
    return out


def _sec(epoch_series):
    return np.floor(epoch_series.astype(float)).astype('int64')


def _read_cols(path, needed):
    """Read only the needed columns that exist (bounds memory on multi-GB
    flow_stats by never loading the unused long 'instructions'/'match' metadata
    columns we don't aggregate)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in needed if c in header]
    return pd.read_csv(path, usecols=cols)


def load_telemetry(telemetry_dir):
    flow = _read_cols(os.path.join(telemetry_dir, 'flow_stats.csv'),
                      ['ts_epoch', 'packet_count', 'byte_count', 'duration_sec', 'match'])
    port = _read_cols(os.path.join(telemetry_dir, 'port_stats.csv'),
                      ['ts_epoch', 'rx_packets', 'tx_packets', 'rx_bytes', 'tx_bytes',
                       'rx_dropped', 'tx_dropped', 'rx_errors', 'tx_errors'])
    table = _read_cols(os.path.join(telemetry_dir, 'table_stats.csv'),
                       ['ts_epoch', 'active_count', 'lookup_count', 'matched_count'])
    for df, name in ((flow, 'flow'), (port, 'port'), (table, 'table')):
        if 'ts_epoch' not in df.columns:
            raise ValueError("%s_stats.csv has no ts_epoch column -- rerun with the v3 telemetry_writer" % name)
        df['sec'] = _sec(df['ts_epoch'])
    return flow, port, table


def parse_flow_match(flow):
    """Vectorized extraction of protocol / addresses / ports from the match string."""
    m = flow['match'].astype(str)
    proto = m.str.extract(r"'ip_proto': (\d+)")[0].astype('float')
    flow['protocol'] = np.select(
        [proto == 6, proto == 17, proto == 1],
        ['tcp', 'udp', 'icmp'], default='other')
    flow['src_ip'] = m.str.extract(r"'ipv4_src': '([^']+)'")[0]
    flow['dst_ip'] = m.str.extract(r"'ipv4_dst': '([^']+)'")[0]
    tcp_src = m.str.extract(r"'tcp_src': (\d+)")[0]
    udp_src = m.str.extract(r"'udp_src': (\d+)")[0]
    tcp_dst = m.str.extract(r"'tcp_dst': (\d+)")[0]
    udp_dst = m.str.extract(r"'udp_dst': (\d+)")[0]
    flow['src_port'] = tcp_src.fillna(udp_src)
    flow['dst_port'] = tcp_dst.fillna(udp_dst)
    return flow


def build_features(flow, port, table, start_sec, end_sec):
    """Return a per-second feature DataFrame indexed by integer epoch-second."""
    full_index = pd.RangeIndex(start_sec, end_sec + 1, name='sec')

    g = flow.groupby('sec')
    feat = pd.DataFrame(index=full_index)
    feat['flow_packet_count_sum'] = g['packet_count'].sum()
    feat['flow_packet_count_mean'] = g['packet_count'].mean()
    feat['flow_packet_count_std'] = g['packet_count'].std()
    feat['flow_byte_count_sum'] = g['byte_count'].sum()
    feat['flow_byte_count_mean'] = g['byte_count'].mean()
    feat['flow_byte_count_std'] = g['byte_count'].std()
    feat['flow_duration_sec_mean'] = g['duration_sec'].mean()
    feat['flow_duration_sec_std'] = g['duration_sec'].std()
    feat['no_of_total_flows'] = g.size()

    proto_counts = flow.groupby(['sec', 'protocol']).size().unstack(fill_value=0)
    for p in ['tcp', 'udp', 'icmp', 'other']:
        feat['no_of_%s_flows' % p] = proto_counts[p] if p in proto_counts else 0
    feat['no_of_unique_src_addresses'] = g['src_ip'].nunique()
    feat['no_of_unique_dst_addresses'] = g['dst_ip'].nunique()
    feat['no_of_unique_src_ports'] = g['src_port'].nunique()
    feat['no_of_unique_dst_ports'] = g['dst_port'].nunique()

    flow_pkt_rowsum = g['packet_count'].sum().reindex(full_index, fill_value=0)
    flow_byte_rowsum = g['byte_count'].sum().reindex(full_index, fill_value=0)
    flow_rowcount = g.size().reindex(full_index, fill_value=0)

    gp = port.groupby('sec')
    for metric in ['rx_packets', 'tx_packets', 'rx_bytes', 'tx_bytes']:
        feat['port_%s_sum' % metric] = gp[metric].sum()
        feat['port_%s_mean' % metric] = gp[metric].mean()
        feat['port_%s_std' % metric] = gp[metric].std()
    feat['port_rx_dropped_sum'] = gp['rx_dropped'].sum()
    feat['port_rx_dropped_mean'] = gp['rx_dropped'].mean()
    feat['port_rx_dropped_std'] = gp['rx_dropped'].std()
    feat['port_tx_errors_sum'] = gp['tx_errors'].sum()
    feat['port_tx_errors_mean'] = gp['tx_errors'].mean()
    feat['port_tx_errors_std'] = gp['tx_errors'].std()
    feat['port_tx_dropped_sum'] = gp['tx_dropped'].sum()
    feat['port_rx_errors_sum'] = gp['rx_errors'].sum()

    port_rx_rowsum = gp['rx_packets'].sum().reindex(full_index, fill_value=0)
    port_tx_rowsum = gp['tx_packets'].sum().reindex(full_index, fill_value=0)
    port_rowcount = gp.size().reindex(full_index, fill_value=0)

    gt = table.groupby('sec')
    for metric in ['active_count', 'lookup_count', 'matched_count']:
        feat['table_%s_sum' % metric] = gt[metric].sum()
        feat['table_%s_mean' % metric] = gt[metric].mean()
        feat['table_%s_std' % metric] = gt[metric].std()

    feat = feat.reindex(full_index)

    feat['tcp_to_udp_ratio'] = feat['no_of_tcp_flows'].fillna(0) / (feat['no_of_udp_flows'].fillna(0) + 1)
    feat['flow_packets_per_flow'] = feat['flow_packet_count_sum'].fillna(0) / (feat['no_of_total_flows'].fillna(0) + 1)
    feat['unique_src_to_dst_address_ratio'] = feat['no_of_unique_src_addresses'].fillna(0) / (feat['no_of_unique_dst_addresses'].fillna(0) + 1)
    feat['unique_src_to_dst_port_ratio'] = feat['no_of_unique_src_ports'].fillna(0) / (feat['no_of_unique_dst_ports'].fillna(0) + 1)

    def roll_mean(rowsum, rowcount):
        rs = rowsum.rolling(10, min_periods=1).sum()
        rc = rowcount.rolling(10, min_periods=1).sum()
        return (rs / rc.replace(0, np.nan)).fillna(0)
    feat['flow_packet_count_10s_mean'] = roll_mean(flow_pkt_rowsum, flow_rowcount)
    feat['flow_byte_count_10s_mean'] = roll_mean(flow_byte_rowsum, flow_rowcount)
    feat['port_rx_packets_10s_mean'] = roll_mean(port_rx_rowsum, port_rowcount)
    feat['port_tx_packets_10s_mean'] = roll_mean(port_tx_rowsum, port_rowcount)
    feat['total_flows_10s'] = flow_rowcount.rolling(10, min_periods=1).sum()

    diff_cols = ['flow_packet_count_sum', 'flow_byte_count_sum', 'port_rx_packets_sum',
                 'port_tx_packets_sum', 'port_rx_bytes_sum', 'port_tx_bytes_sum',
                 'no_of_total_flows', 'no_of_tcp_flows', 'no_of_udp_flows',
                 'no_of_icmp_flows', 'no_of_other_flows', 'no_of_unique_src_addresses']
    for c in diff_cols:
        feat['%s_diff' % c] = feat[c].fillna(0).diff().fillna(0)

    feat = feat.fillna(0)
    feat.index.name = 'sec'
    return feat.reset_index()


def assign_ground_truth(secs, windows):
    """For each integer second, assign (window_id, kind, type) by which window
    covers the second's midpoint.

    window_type preserves the benign subtype from the sim log ('regular' vs
    'cooldown') so the cooldown guard band can be excluded downstream; attack
    seconds carry the attack label. Seconds not covered by any window (tiny
    sub-second boundary gaps between consecutive windows) are marked 'gap' and
    excluded alongside the cooldown, since they straddle a transition."""
    starts = np.array([w['start_epoch'] for w in windows])
    n = len(secs)
    win_id = np.zeros(n, dtype='int64')
    win_kind = np.array(['normal'] * n, dtype=object)
    win_type = np.array(['gap'] * n, dtype=object)

    filler_base = max(w['id'] for w in windows) + 1
    mid = secs.astype(float) + 0.5
    idx = np.searchsorted(starts, mid, side='right') - 1
    for i in range(n):
        j = idx[i]
        if j >= 0 and windows[j]['start_epoch'] <= mid[i] < windows[j]['end_epoch']:
            win_id[i] = windows[j]['id']
            win_kind[i] = windows[j]['kind']
            win_type[i] = windows[j]['type']
        else:
            win_id[i] = filler_base + i
            win_kind[i] = 'normal'
            win_type[i] = 'gap'
    return win_id, win_kind, win_type


def snort_labels_per_sec(secs, alerts):
    """For each second, return the dominant Snort attack label (or None)."""
    by_sec = {}
    for epoch, atype in alerts:
        s = int(np.floor(epoch))
        by_sec.setdefault(s, {})
        by_sec[s][atype] = by_sec[s].get(atype, 0) + 1
    labels = []
    for s in secs:
        counts = by_sec.get(int(s))
        if not counts:
            labels.append(None)
        else:
            labels.append(max(counts.items(), key=lambda kv: kv[1])[0])
    return labels


def one_hot(df, label_array):
    df['attack_syn'] = (label_array == 'syn').astype(int)
    df['attack_udp'] = (label_array == 'udp').astype(int)
    df['attack_icmp'] = (label_array == 'icmp').astype(int)
    df['attack_goldeneye'] = (label_array == 'goldeneye').astype(int)
    df['attack'] = df[['attack_syn', 'attack_udp', 'attack_icmp', 'attack_goldeneye']].max(axis=1)
    return df


def snort_coverage(gt_type, snort_label):
    """Per-type Snort second-level coverage vs ground truth (incl. GoldenEye)."""
    print("\nSnort coverage vs ground truth (second-level):")
    for t in ATTACK_LABELS:
        gt_secs = (gt_type == t)
        total = int(gt_secs.sum())
        detected = int(((snort_label == t) & gt_secs).sum())
        cov = (detected / total * 100) if total else 0.0
        print("  %-10s ground-truth=%5d  snort-detected=%5d  coverage=%.1f%%" % (t, total, detected, cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--telemetry-dir', required=True,
                    help='Dir with flow_stats.csv/port_stats.csv/table_stats.csv (and usually the artifacts)')
    ap.add_argument('--alerts', default=None, help='Snort alerts file (default: <telemetry-dir>/alerts.csv)')
    ap.add_argument('--sim-log', default=None, help='traffic_simulation.log (default: <telemetry-dir>/traffic_simulation.log)')
    ap.add_argument('--out-dir', default=None, help='Where to write the datasets (default: telemetry-dir)')
    ap.add_argument('--max-skew', type=float, default=2.0,
                    help='Abort if telemetry does not cover the sim window within this many seconds')
    ap.add_argument('--allow-skew', action='store_true', help='Warn instead of aborting on skew/coverage issues')
    ap.add_argument('--keep-cooldown', action='store_true',
                    help='Keep cooldown/boundary-gap rows in the ML datasets (default: exclude them as a '
                         'guard band and write them to cooldown_rows.csv for the idle-timeout-echo study)')
    ap.add_argument('--keep-empty', action='store_true',
                    help='Keep empty (no-telemetry-poll) rows in the ML datasets (default: exclude them). '
                         'These are all-zero rows for seconds where NO flow/port/table poll landed -- the 1 Hz '
                         'poller stalls under flood load (large multipart flow-stats replies), so whole seconds '
                         'are skipped and reindex-filled with 0. They carry a label but no signal and create '
                         'spurious first-difference spikes, so they are dropped and written to empty_rows.csv '
                         'for row accounting (the main reason recorded-seconds > dataset-rows).')
    args = ap.parse_args()

    tdir = args.telemetry_dir
    alerts_path = args.alerts or os.path.join(tdir, 'alerts.csv')
    sim_log_path = args.sim_log or os.path.join(tdir, 'traffic_simulation.log')
    out_dir = args.out_dir or tdir
    os.makedirs(out_dir, exist_ok=True)

    print("Parsing simulation log:", sim_log_path)
    run_start, run_end, windows = parse_sim_log(sim_log_path)
    n_attack = sum(1 for w in windows if w['kind'] == 'attack')
    print("  run window: %s -> %s" % (datetime.fromtimestamp(run_start, timezone.utc).isoformat(),
                                       datetime.fromtimestamp(run_end, timezone.utc).isoformat()))
    print("  windows: %d total, %d attack, %d benign" % (len(windows), n_attack, len(windows) - n_attack))

    print("Parsing Snort alerts:", alerts_path)
    alerts = parse_alerts(alerts_path, run_start, run_end) if os.path.exists(alerts_path) else []
    print("  parsed %d attack alerts" % len(alerts))

    print("Loading telemetry from:", tdir)
    flow, port, table = load_telemetry(tdir)
    flow = parse_flow_match(flow)

    tel_min, tel_max = int(flow['sec'].min()), int(flow['sec'].max())
    start_sec, end_sec = int(np.floor(run_start)), int(np.ceil(run_end))

    gap_start = tel_min - start_sec
    gap_end = end_sec - tel_max
    print("\nTime alignment: telemetry [%d..%d] vs sim [%d..%d] (lead=%ds, lag=%ds)" % (
        tel_min, tel_max, start_sec, end_sec, gap_start, gap_end))
    if (tel_min > start_sec + args.max_skew) or (tel_max < end_sec - args.max_skew):
        msg = ("Telemetry does not cover the simulation window within %.1fs. "
               "This usually means the VM and host clocks drifted -- re-run "
               "tools/sync_clocks.py before the next capture." % args.max_skew)
        if args.allow_skew:
            print("WARNING:", msg)
        else:
            print("ABORT:", msg, "\n(use --allow-skew to force.)")
            sys.exit(2)

    start_sec = max(start_sec, tel_min)
    end_sec = min(end_sec, tel_max)

    print("Building features for seconds %d..%d (%d rows)" % (start_sec, end_sec, end_sec - start_sec + 1))
    feat = build_features(flow, port, table, start_sec, end_sec)
    secs = feat['sec'].values
    feat = feat.drop(columns=['sec'])

    feat.insert(0, 'ts_epoch', secs.astype(float))
    feat.insert(0, 'timestamp', [datetime.fromtimestamp(s, timezone.utc).isoformat() for s in secs])

    win_id, win_kind, win_type = assign_ground_truth(secs, windows)
    feat['window_id'] = win_id
    feat['window_kind'] = win_kind
    feat['window_type'] = win_type

    gt_label = np.array(win_type, dtype=object)
    gt_label[(win_kind != 'attack')] = 'normal'

    snort_label = np.array(snort_labels_per_sec(secs, alerts), dtype=object)

    snort_coverage(gt_label, snort_label)

    corrected = feat.copy()
    corrected = one_hot(corrected, gt_label)

    original = feat.copy()
    original = one_hot(original, snort_label)

    present_secs = np.fromiter(
        set(flow['sec'].unique()) | set(port['sec'].unique()) | set(table['sec'].unique()),
        dtype='int64')
    empty_mask = ~np.isin(secs.astype('int64'), present_secs)
    n_empty = int(empty_mask.sum())

    guard_mask = feat['window_type'].isin(['cooldown', 'gap']).values
    n_guard = int(guard_mask.sum())

    drop_mask = np.zeros(len(feat), dtype=bool)
    if not args.keep_empty:
        if n_empty:
            corrected.loc[empty_mask].to_csv(os.path.join(out_dir, 'empty_rows.csv'), index=False)
        drop_mask |= empty_mask
    if not args.keep_cooldown:
        corrected.loc[guard_mask & ~empty_mask].to_csv(os.path.join(out_dir, 'cooldown_rows.csv'), index=False)
        drop_mask |= guard_mask

    corrected = corrected.loc[~drop_mask].reset_index(drop=True)
    original = original.loc[~drop_mask].reset_index(drop=True)
    print("\nRow exclusions:")
    print("  empty (no-telemetry-poll) rows: %d  %s"
          % (n_empty, "KEPT (--keep-empty)" if args.keep_empty else "-> empty_rows.csv"))
    print("  cooldown/gap guard-band rows:   %d  %s"
          % (n_guard, "KEPT (--keep-cooldown)" if args.keep_cooldown else "-> cooldown_rows.csv (non-empty only)"))
    print("  total dropped: %d of %d rows (%d retained)"
          % (int(drop_mask.sum()), len(feat), len(feat) - int(drop_mask.sum())))

    out_c = os.path.join(out_dir, 'dataset_v3-corrected.csv')
    out_o = os.path.join(out_dir, 'dataset_v3.csv')
    corrected.to_csv(out_c, index=False)
    original.to_csv(out_o, index=False)

    print("\n=== label summary ===")
    for name, d in (('corrected (ground truth)', corrected), ('original (Snort)', original)):
        print("  %-26s attack=%d  syn=%d udp=%d icmp=%d goldeneye=%d  normal=%d" % (
            name, int(d['attack'].sum()), int(d['attack_syn'].sum()), int(d['attack_udp'].sum()),
            int(d['attack_icmp'].sum()), int(d['attack_goldeneye'].sum()), int((d['attack'] == 0).sum())))
    print("\nWrote:\n  %s\n  %s" % (out_c, out_o))
    print("Features: %d" % (len([c for c in corrected.columns if c not in
          ['timestamp', 'ts_epoch', 'window_id', 'window_kind', 'window_type',
           'attack', 'attack_syn', 'attack_udp', 'attack_icmp', 'attack_goldeneye']])))


if __name__ == '__main__':
    main()
