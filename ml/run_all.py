#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, extra):
    cmd = [sys.executable, os.path.join(HERE, script)] + extra
    print("\n$ " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("!! %s exited with %d" % (script, rc))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='Capture dir (telemetry + artifacts + datasets)')
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--allow-skew', action='store_true')
    args = ap.parse_args()

    corrected = os.path.join(args.run_dir, 'dataset_v3-corrected.csv')
    original = os.path.join(args.run_dir, 'dataset_v3.csv')

    if not args.skip_generate:
        gen = ['--telemetry-dir', args.run_dir]
        if args.allow_skew:
            gen.append('--allow-skew')
        if run('generate_dataset.py', gen) != 0:
            print("Dataset generation failed; stopping."); return 1

    run('evaluate_windowgrouped.py', ['--data', corrected, '--out', os.path.join(args.run_dir, 'windowgrouped_results')])
    run('evaluate_random_split.py', ['--data', corrected, '--out', os.path.join(args.run_dir, 'random_split_results')])
    run('evaluate_common_testset.py', ['--original', original, '--corrected', corrected,
                                        '--out', os.path.join(args.run_dir, 'common_testset_results')])
    run('plots.py', ['--run-dir', args.run_dir,
                     '--windowgrouped', os.path.join(args.run_dir, 'windowgrouped_results'),
                     '--random', os.path.join(args.run_dir, 'random_split_results'),
                     '--common', os.path.join(args.run_dir, 'common_testset_results'),
                     '--out', os.path.join(args.run_dir, 'figures')])
    print("\n=== v3 ML pipeline complete; results under", args.run_dir, "===")
    return 0


if __name__ == '__main__':
    sys.exit(main())
