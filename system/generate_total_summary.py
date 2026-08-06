"""
generate_total_summary.py — Generate Total Summary Tables for Each Experiment Group
====================================================================================

For each of the 5 experiment groups, generates a total_summary_<group>.txt file
containing tables with:
  - Best Skewed Accuracy (max across ALL rounds) and its round number
  - Best Whole-Label Accuracy (max across ALL rounds) and its round number
  - Final Round Skewed Accuracy
  - Final Round Whole-Label Accuracy

Best accuracies are selected by the HIGHEST individual metric value,
NOT by monitoring score.

USAGE:
    python3 generate_total_summary.py

OUTPUT:
    results/fedsala_75_vs_100/total_summary_fedsala_75_vs_100.txt
    results/fedsala_all_experiments_all_four_groups/group1_cifar10_baseline/total_summary_group1_cifar10_baseline.txt
    results/fedsala_all_experiments_all_four_groups/group2_cifar100_baseline/total_summary_group2_cifar100_baseline.txt
    results/fedsala_all_experiments_all_four_groups/group3_fedala_l_comparison/total_summary_group3_fedala_l_comparison.txt
    results/fedsala_all_experiments_all_four_groups/group4_fedsala_g_comparison/total_summary_group4_fedsala_g_comparison.txt
"""

import os
import sys
import numpy as np
from datetime import datetime


# =========================================================================
# Configuration for each group
# =========================================================================
# Each group is defined by:
#   - group_dir: path to the group results folder
#   - group_name: display name
#   - splits: list of split IDs (e.g. S2, S3, ...)
#   - methods: list of (display_label, folder_suffix, algo_prefix) tuples
#     where folder_suffix is used to find the dir: {split}_{folder_suffix}
#     and algo_prefix is the .npy file prefix (e.g. FedSALA, FedALA, etc.)

RESULTS_BASE = os.path.join('..', 'results')
FOUR_GROUPS_BASE = os.path.join(RESULTS_BASE, 'fedsala_all_experiments_all_four_groups')

CIFAR10_SPLITS = ['S2', 'S3', 'S4', 'SC', 'SD', 'SE', 'SF', 'SA2', 'SA4', 'SMX']
CIFAR100_SPLITS = ['C100_5c', 'C100_10c', 'C100_20c']

GROUPS = [
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group1_cifar10_baseline'),
        'group_name': 'Group 1 — CIFAR-10 Baseline (FedSALA vs FedALA vs FedAvg vs LocalOnly)',
        'splits': CIFAR10_SPLITS,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedALA',     'FedALA',     'FedALA'),
            ('FedAvg',     'FedAvg',     'FedAvg'),
            ('LocalOnly',  'LocalOnly',  'LocalOnly'),
        ],
    },
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group2_cifar100_baseline'),
        'group_name': 'Group 2 — CIFAR-100 Baseline (FedSALA vs FedALA vs FedAvg vs LocalOnly)',
        'splits': CIFAR100_SPLITS,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedALA',     'FedALA',     'FedALA'),
            ('FedAvg',     'FedAvg',     'FedAvg'),
            ('LocalOnly',  'LocalOnly',  'LocalOnly'),
        ],
    },
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group3_fedala_l_comparison'),
        'group_name': 'Group 3 — FedSALA vs FedALA-L (Lower Layers Local)',
        'splits': CIFAR10_SPLITS,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedALA-L',   'FedALA-L',   'FedALA'),
        ],
    },
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group4_fedsala_g_comparison'),
        'group_name': 'Group 4 — FedSALA vs FedSALA-G (Global Objective)',
        'splits': CIFAR10_SPLITS,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedSALA-G',  'FedSALA-G',  'FedSALA'),
        ],
    },
    {
        'group_dir': os.path.join(RESULTS_BASE, 'fedsala_75_vs_100'),
        'group_name': 'Group 5 — FedSALA-75 vs FedSALA-100 (Threshold Ablation)',
        'splits': CIFAR10_SPLITS,
        'methods': [
            ('FedSALA-75',  'FedSALA-75',  'FedSALA'),
            ('FedSALA-100', 'FedSALA-100', 'FedSALA'),
        ],
    },
]


# =========================================================================
# Data Loading Helpers
# =========================================================================

def load_npy(path):
    """Load a .npy file if it exists, return None otherwise."""
    if os.path.exists(path):
        return np.load(path)
    return None


def get_skewed_acc(result_dir, algo_prefix):
    """Load skewed/local accuracy array.
    FedALA/FedSALA use 'post_local_acc', FedAvg/LocalOnly use 'local_acc'.
    """
    data = load_npy(os.path.join(result_dir, f'{algo_prefix}_post_local_acc.npy'))
    if data is not None:
        return data
    data = load_npy(os.path.join(result_dir, f'{algo_prefix}_local_acc.npy'))
    return data


def get_global_acc(result_dir, algo_prefix):
    """Load whole-label/global accuracy array.
    FedALA/FedSALA use 'pre_global_acc', FedAvg/LocalOnly use 'global_acc'.
    """
    data = load_npy(os.path.join(result_dir, f'{algo_prefix}_pre_global_acc.npy'))
    if data is not None:
        return data
    data = load_npy(os.path.join(result_dir, f'{algo_prefix}_global_acc.npy'))
    return data


# =========================================================================
# Summary Generation
# =========================================================================

def generate_group_summary(group_config):
    """Generate a total summary text file for one experiment group."""

    group_dir = group_config['group_dir']
    group_name = group_config['group_name']
    splits = group_config['splits']
    methods = group_config['methods']

    if not os.path.isdir(group_dir):
        print(f"  WARNING: {group_dir} not found, skipping.")
        return

    # Determine output filename from the last component of group_dir
    dir_basename = os.path.basename(group_dir)
    out_path = os.path.join(group_dir, f'total_summary_{dir_basename}.txt')

    lines = []
    W = 120  # line width

    lines.append('=' * W)
    lines.append(f'  TOTAL SUMMARY: {group_name}')
    lines.append(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * W)
    lines.append('')
    lines.append('  NOTE: "Best" accuracies are selected by the HIGHEST individual metric')
    lines.append('        value across ALL communication rounds — NOT by monitoring score.')
    lines.append('')

    # Method display labels
    method_labels = [m[0] for m in methods]

    # ----------------------------------------------------------------
    # TABLE 1: Best Skewed Local Test Accuracy
    # ----------------------------------------------------------------
    lines.append('-' * W)
    lines.append('  TABLE 1: Best Skewed Local Test Accuracy (max across all rounds)')
    lines.append('-' * W)

    # Build header
    hdr_parts = [f'{"Split":<8}']
    for label in method_labels:
        hdr_parts.append(f'{"Best Acc":>10}')
        hdr_parts.append(f'{"Round":>6}')
        hdr_parts.append(f'{"Final Acc":>10}')
        hdr_parts.append(f' |')
    header = '  ' + '  '.join(hdr_parts)

    # Method name header row
    mhdr_parts = [f'{"":8}']
    for label in method_labels:
        col_w = 10 + 6 + 10 + 2 + 6  # widths of Best Acc + Round + Final Acc + separators
        mhdr_parts.append(f'{label:^{col_w}}')
    lines.append('  ' + '  '.join(mhdr_parts))
    lines.append(header)
    lines.append('  ' + '-' * (W - 2))

    for split_id in splits:
        row_parts = [f'{split_id:<8}']
        for label, folder_suffix, algo_prefix in methods:
            result_dir = os.path.join(group_dir, f'{split_id}_{folder_suffix}')
            skewed = get_skewed_acc(result_dir, algo_prefix)
            if skewed is not None and len(skewed) > 0:
                best_idx = int(np.argmax(skewed))
                best_val = skewed[best_idx]
                final_val = skewed[-1]
                row_parts.append(f'{best_val:>10.4f}')
                row_parts.append(f'{best_idx:>6d}')
                row_parts.append(f'{final_val:>10.4f}')
            else:
                row_parts.append(f'{"N/A":>10}')
                row_parts.append(f'{"N/A":>6}')
                row_parts.append(f'{"N/A":>10}')
            row_parts.append(f' |')
        lines.append('  ' + '  '.join(row_parts))

    lines.append('')
    lines.append('')

    # ----------------------------------------------------------------
    # TABLE 2: Best Whole-Label (Global) Test Accuracy
    # ----------------------------------------------------------------
    lines.append('-' * W)
    lines.append('  TABLE 2: Best Whole-Label Test Accuracy (max across all rounds)')
    lines.append('-' * W)

    # Reuse same header format
    lines.append('  ' + '  '.join(mhdr_parts))
    lines.append(header)
    lines.append('  ' + '-' * (W - 2))

    for split_id in splits:
        row_parts = [f'{split_id:<8}']
        for label, folder_suffix, algo_prefix in methods:
            result_dir = os.path.join(group_dir, f'{split_id}_{folder_suffix}')
            global_acc = get_global_acc(result_dir, algo_prefix)
            if global_acc is not None and len(global_acc) > 0:
                best_idx = int(np.argmax(global_acc))
                best_val = global_acc[best_idx]
                final_val = global_acc[-1]
                row_parts.append(f'{best_val:>10.4f}')
                row_parts.append(f'{best_idx:>6d}')
                row_parts.append(f'{final_val:>10.4f}')
            else:
                row_parts.append(f'{"N/A":>10}')
                row_parts.append(f'{"N/A":>6}')
                row_parts.append(f'{"N/A":>10}')
            row_parts.append(f' |')
        lines.append('  ' + '  '.join(row_parts))

    lines.append('')
    lines.append('')

    # ----------------------------------------------------------------
    # TABLE 3: Combined Overview (Skewed + Whole-Label side by side)
    # ----------------------------------------------------------------
    lines.append('-' * W)
    lines.append('  TABLE 3: Combined Overview — Best Accuracy per Method per Split')
    lines.append('-' * W)
    lines.append('')

    for split_id in splits:
        lines.append(f'  ┌─ {split_id} ' + '─' * (W - 8))

        sub_hdr = f'  │ {"Method":<15} │ {"Best Skew":>10} {"@Rnd":>5} │ {"Final Skew":>10} │ {"Best WL":>10} {"@Rnd":>5} │ {"Final WL":>10} │'
        lines.append(sub_hdr)
        lines.append(f'  │ {"-"*15} │ {"-"*10} {"-"*5} │ {"-"*10} │ {"-"*10} {"-"*5} │ {"-"*10} │')

        for label, folder_suffix, algo_prefix in methods:
            result_dir = os.path.join(group_dir, f'{split_id}_{folder_suffix}')
            skewed = get_skewed_acc(result_dir, algo_prefix)
            global_acc = get_global_acc(result_dir, algo_prefix)

            if skewed is not None and len(skewed) > 0:
                sk_best_idx = int(np.argmax(skewed))
                sk_best = f'{skewed[sk_best_idx]:.4f}'
                sk_rnd = f'{sk_best_idx}'
                sk_final = f'{skewed[-1]:.4f}'
            else:
                sk_best, sk_rnd, sk_final = 'N/A', 'N/A', 'N/A'

            if global_acc is not None and len(global_acc) > 0:
                gl_best_idx = int(np.argmax(global_acc))
                gl_best = f'{global_acc[gl_best_idx]:.4f}'
                gl_rnd = f'{gl_best_idx}'
                gl_final = f'{global_acc[-1]:.4f}'
            else:
                gl_best, gl_rnd, gl_final = 'N/A', 'N/A', 'N/A'

            row = f'  │ {label:<15} │ {sk_best:>10} {sk_rnd:>5} │ {sk_final:>10} │ {gl_best:>10} {gl_rnd:>5} │ {gl_final:>10} │'
            lines.append(row)

        lines.append(f'  └' + '─' * (W - 3))
        lines.append('')

    lines.append('=' * W)

    # Write to file
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  ✓ {out_path}")


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    print("\n  Generating total summary tables...\n")
    for group in GROUPS:
        generate_group_summary(group)
    print("\n  All summaries generated.\n")
