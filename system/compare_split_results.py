"""
compare_split_results.py — Generate Combined Comparison Plots for a Split
==========================================================================

Takes multiple result directories (one per method) for the same data split
and generates combined comparison plots showing all methods side-by-side.

USAGE:
    python3 compare_split_results.py \
        --split_id S2 \
        --group_dir results/fedsala_all_experiments/group1_cifar10_baseline \
        --methods LocalOnly FedAvg FedALA FedSALA \
        --eval_gap 1

OUTPUT:
    <group_dir>/S2_comparison/
        ├── comparison_skewed.png
        ├── comparison_global.png
        ├── comparison_loss.png
        └── comparison_summary.txt
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Consistent color scheme across all plots
COLORS = {
    'FedSALA':      '#e74c3c',   # Red (our method — highlight)
    'FedSALA-75':   '#e74c3c',   # Red (75.16% threshold variant)
    'FedSALA-100':  '#3498db',   # Blue (100% threshold variant)
    'FedALA':       '#3498db',   # Blue
    'FedAvg':       '#2ecc71',   # Green
    'LocalOnly':    '#9b59b6',   # Purple
    'FedALA-L':     '#f39c12',   # Orange (FedALA with lower layers local)
    'FedSALA-L':    '#f39c12',   # Orange (fallback)
    'FedSALA-G':    '#1abc9c',   # Teal
}

LINEWIDTHS = {
    'FedSALA':      3.0,
    'FedSALA-75':   3.0,
    'FedSALA-100':  1.5,
    'FedALA':       1.5,
    'FedAvg':       1.5,
    'LocalOnly':    1.5,
    'FedALA-L':     1.5,
    'FedSALA-L':    1.5,
    'FedSALA-G':    1.5,
}


def load_metric(result_dir, algo_name, metric_name):
    """Try to load a .npy metric file from a result directory."""
    path = os.path.join(result_dir, f'{algo_name}_{metric_name}.npy')
    if os.path.exists(path):
        return np.load(path)
    return None


def get_skewed_acc(result_dir, algo_name):
    """Load skewed/local accuracy — different key names for different methods."""
    # FedALA/FedSALA use 'post_local_acc'
    data = load_metric(result_dir, algo_name, 'post_local_acc')
    if data is not None:
        return data
    # FedAvg/LocalOnly use 'local_acc'
    data = load_metric(result_dir, algo_name, 'local_acc')
    return data


def get_global_acc(result_dir, algo_name):
    """Load global/whole-label accuracy — different key names for different methods."""
    # FedALA/FedSALA use 'pre_global_acc'
    data = load_metric(result_dir, algo_name, 'pre_global_acc')
    if data is not None:
        return data
    # FedAvg/LocalOnly use 'global_acc'
    data = load_metric(result_dir, algo_name, 'global_acc')
    return data


def get_train_loss(result_dir, algo_name):
    """Load training loss."""
    return load_metric(result_dir, algo_name, 'train_loss')


def get_monitoring_score(result_dir, algo_name):
    """Load monitoring score."""
    return load_metric(result_dir, algo_name, 'monitoring_score')


def make_comparison_plots(split_id, group_dir, methods, eval_gap, algo_names=None):
    """Generate combined comparison plots for all methods on one split.
    
    Args:
        methods: Display label names used in folder names (e.g. FedSALA-L)
        algo_names: Internal algorithm names used as .npy file prefixes (e.g. FedALA).
                    If None, same as methods.
    """

    if algo_names is None:
        algo_names = methods

    out_dir = os.path.join(group_dir, f'{split_id}_comparison')
    os.makedirs(out_dir, exist_ok=True)

    # Collect data for each method
    all_data = {}
    for method, algo_name in zip(methods, algo_names):
        result_dir = os.path.join(group_dir, f'{split_id}_{method}')
        if not os.path.isdir(result_dir):
            print(f"  WARNING: {result_dir} not found, skipping {method}")
            continue

        all_data[method] = {
            'skewed': get_skewed_acc(result_dir, algo_name),
            'global': get_global_acc(result_dir, algo_name),
            'loss':   get_train_loss(result_dir, algo_name),
            'score':  get_monitoring_score(result_dir, algo_name),
        }

    if not all_data:
        print(f"  ERROR: No data found for split {split_id}")
        return

    def get_rounds(metric_arr):
        if metric_arr is None:
            return []
        return list(range(0, len(metric_arr) * eval_gap, eval_gap))[:len(metric_arr)]

    # ---- PLOT 1: Skewed Local Test Accuracy ----
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    for method, data in all_data.items():
        arr = data['skewed']
        if arr is not None and len(arr) > 0:
            rounds = get_rounds(arr)
            plt.plot(rounds, arr, label=method,
                     color=COLORS.get(method, '#95a5a6'),
                     linewidth=LINEWIDTHS.get(method, 1.5),
                     alpha=0.9)
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Test Accuracy (Skewed Local)', fontsize=14)
    plt.title(f'{split_id} — Skewed Local Test Accuracy', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_skewed.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path}")

    # ---- PLOT 2: Whole-Label (Global) Test Accuracy ----
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    for method, data in all_data.items():
        arr = data['global']
        if arr is not None and len(arr) > 0:
            rounds = get_rounds(arr)
            plt.plot(rounds, arr, label=method,
                     color=COLORS.get(method, '#95a5a6'),
                     linewidth=LINEWIDTHS.get(method, 1.5),
                     alpha=0.9)
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Test Accuracy (Whole-Label)', fontsize=14)
    plt.title(f'{split_id} — Whole-Label Test Accuracy', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_global.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path}")

    # ---- PLOT 3: Training Loss ----
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    for method, data in all_data.items():
        arr = data['loss']
        if arr is not None and len(arr) > 0:
            rounds = get_rounds(arr)
            plt.plot(rounds, arr, label=method,
                     color=COLORS.get(method, '#95a5a6'),
                     linewidth=LINEWIDTHS.get(method, 1.5),
                     alpha=0.9)
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.title(f'{split_id} — Training Loss', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_loss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {path}")

    # ---- Summary text file ----
    summary_path = os.path.join(out_dir, 'comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"  COMPARISON SUMMARY: {split_id}\n")
        f.write(f"{'='*80}\n\n")
        
        # 1. Final Round Table
        f.write("  --- FINAL ROUND RESULTS ---\n")
        f.write(f"  {'Method':<15} {'Skewed Acc':<15} {'Global Acc':<15} {'Loss':<12} {'Score':<10}\n")
        f.write(f"  {'-'*70}\n")
        for method, data in all_data.items():
            skewed_final = f"{data['skewed'][-1]:.4f}" if data['skewed'] is not None and len(data['skewed']) > 0 else "N/A"
            global_final = f"{data['global'][-1]:.4f}" if data['global'] is not None and len(data['global']) > 0 else "N/A"
            loss_final = f"{data['loss'][-1]:.4f}" if data['loss'] is not None and len(data['loss']) > 0 else "N/A"
            score_final = f"{data['score'][-1]:.4f}" if data['score'] is not None and len(data['score']) > 0 else "N/A"
            f.write(f"  {method:<15} {skewed_final:<15} {global_final:<15} {loss_final:<12} {score_final:<10}\n")
            
        f.write("\n")
        
        # 2. Best Round Table
        f.write("  --- BEST ROUND RESULTS (Maximized Monitoring Score) ---\n")
        f.write(f"  {'Method':<15} {'Best Round':<12} {'Skewed Acc':<15} {'Global Acc':<15} {'Loss':<12} {'Score':<10}\n")
        f.write(f"  {'-'*80}\n")
        for method, data in all_data.items():
            if data['score'] is not None and len(data['score']) > 0:
                best_idx = int(np.argmax(data['score']))
                best_skewed = f"{data['skewed'][best_idx]:.4f}" if data['skewed'] is not None and len(data['skewed']) > best_idx else "N/A"
                best_global = f"{data['global'][best_idx]:.4f}" if data['global'] is not None and len(data['global']) > best_idx else "N/A"
                best_loss = f"{data['loss'][best_idx]:.4f}" if data['loss'] is not None and len(data['loss']) > best_idx else "N/A"
                best_score = f"{data['score'][best_idx]:.4f}"
                best_round_str = str(best_idx * eval_gap)
            else:
                best_skewed = "N/A"
                best_global = "N/A"
                best_loss = "N/A"
                best_score = "N/A"
                best_round_str = "N/A"
            f.write(f"  {method:<15} {best_round_str:<12} {best_skewed:<15} {best_global:<15} {best_loss:<12} {best_score:<10}\n")
            
        f.write(f"\n{'='*80}\n")
    print(f"  Summary: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate combined comparison plots for a split')
    parser.add_argument('--split_id', type=str, required=True, help='Split ID (e.g. S2)')
    parser.add_argument('--group_dir', type=str, required=True, help='Group results directory')
    parser.add_argument('--methods', nargs='+', required=True, help='Method display names (e.g. LocalOnly FedAvg FedALA FedSALA)')
    parser.add_argument('--algo_names', nargs='+', default=None, help='Internal algo names for .npy files (e.g. FedALA FedSALA). If omitted, uses --methods.')
    parser.add_argument('--eval_gap', type=int, default=1, help='Evaluation gap (default: 1)')
    args = parser.parse_args()

    make_comparison_plots(args.split_id, args.group_dir, args.methods, args.eval_gap, args.algo_names)
