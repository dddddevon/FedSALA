"""
generate_experiment_results_latex.py — Generate LaTeX file with all experiment results
======================================================================================
Reads .npy result files and comparison graphs from all 5 experiment groups and
generates a single comprehensive LaTeX file.
"""

import os
import sys
import numpy as np
from datetime import datetime

# =========================================================================
# Paths
# =========================================================================
RESULTS_BASE = os.path.join('..', 'results')
FOUR_GROUPS_BASE = os.path.join(RESULTS_BASE, 'fedsala_all_experiments_all_four_groups')

# Absolute paths for LaTeX \includegraphics
ABS_RESULTS = os.path.abspath(RESULTS_BASE)
ABS_FOUR_GROUPS = os.path.abspath(FOUR_GROUPS_BASE)

OUTPUT_DIR = os.path.abspath('../latex_output')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'experiment_results_all_groups.tex')

CIFAR10_SPLITS = ['S2', 'S3', 'S4', 'SC', 'SD', 'SE', 'SF', 'SA2', 'SA4', 'SMX']
CIFAR100_SPLITS = ['C100_5c', 'C100_10c', 'C100_20c']

# Split descriptions for the data-split table
CIFAR10_SPLIT_DESC = {
    'S2':  ('comp\\_2label',         10, '2 labels/client, equal ratio [1:1], 400 samples'),
    'S3':  ('comp\\_3label',         10, '3 labels/client, equal ratio [1:1:1], 600 samples'),
    'S4':  ('comp\\_4label',         10, '4 labels/client, equal ratio [1:1:1:1], 800 samples'),
    'SC':  ('remote\\_sensing',      10, '3 labels/client, skewed ratio [4:2:1], 560 samples'),
    'SD':  ('hospital\\_uniform',    10, '3 labels/client, skewed ratio [3:2:1], 600 samples'),
    'SE':  ('hospital\\_mixed',      10, '4 clients: 4 labels [4:3:2:1], 6 clients: 2 labels [3:1]'),
    'SF':  ('camera\\_trap',         15, '2 labels/client, extreme skew [6:1], 1260 samples'),
    'SA2': ('app\\_2label\\_skewed',  15, '2 labels/client, skewed ratio [3:1], 400 samples'),
    'SA4': ('app\\_4label\\_skewed',   5, '4 labels/client, skewed ratio [4:3:2:1], 800 samples'),
    'SMX': ('app\\_mixed\\_hetero',   10, '5 clients: 2 labels [3:1], 5 clients: 4 labels [4:3:2:1]'),
}

CIFAR100_SPLIT_DESC = {
    'C100_5c':  ('pathological\\_5c',   20, '5 classes/client, 500 samples/label (extreme)'),
    'C100_10c': ('pathological\\_10c',  20, '10 classes/client, 250 samples/label (standard)'),
    'C100_20c': ('pathological\\_20c',  20, '20 classes/client, 125 samples/label (mild)'),
}

# =========================================================================
# Data Loading
# =========================================================================
def load_npy(path):
    if os.path.exists(path):
        return np.load(path)
    return None

def get_skewed_acc(result_dir, algo_prefix):
    data = load_npy(os.path.join(result_dir, f'{algo_prefix}_post_local_acc.npy'))
    if data is not None:
        return data
    return load_npy(os.path.join(result_dir, f'{algo_prefix}_local_acc.npy'))

def get_global_acc(result_dir, algo_prefix):
    data = load_npy(os.path.join(result_dir, f'{algo_prefix}_pre_global_acc.npy'))
    if data is not None:
        return data
    return load_npy(os.path.join(result_dir, f'{algo_prefix}_global_acc.npy'))

def get_stats(arr):
    """Return (best_val, best_round, final_val) or (None,None,None)."""
    if arr is not None and len(arr) > 0:
        best_idx = int(np.argmax(arr))
        return float(arr[best_idx]), best_idx, float(arr[-1])
    return None, None, None

def fmt(val, decimals=4):
    if val is None:
        return 'N/A'
    return f'{val:.{decimals}f}'

def fmt_rnd(val):
    if val is None:
        return 'N/A'
    return str(val)

def bold_best(values, idx):
    """Return formatted string, bolding the best (max) value."""
    vals = [v for v in values if v is not None]
    if not vals:
        return fmt(values[idx])
    best = max(vals)
    if values[idx] is not None and values[idx] == best:
        return '\\textbf{' + fmt(values[idx]) + '}'
    return fmt(values[idx])

# =========================================================================
# LaTeX Generation
# =========================================================================

GROUPS = [
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group1_cifar10_baseline'),
        'group_num': 1,
        'group_title': 'CIFAR-10 Baseline Comparison',
        'group_desc': 'Compares FedSALA against FedALA, FedAvg, and LocalOnly across 10 non-IID CIFAR-10 data splits.',
        'dataset': 'CIFAR-10',
        'model': 'ResNet-18',
        'rounds': 200,
        'lr': 0.001,
        'batch_size': 10,
        'extra_setup': 'FedSALA: Method 3, $\\tau=0.7516$, EMA$=0.5$, sample$=10\\%$. FedALA: layer\\_idx$=17$, $\\eta=1.0$, rand$=80\\%$.',
        'splits': CIFAR10_SPLITS,
        'split_desc': CIFAR10_SPLIT_DESC,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedALA',     'FedALA',     'FedALA'),
            ('FedAvg',     'FedAvg',     'FedAvg'),
            ('LocalOnly',  'LocalOnly',  'LocalOnly'),
        ],
    },
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group2_cifar100_baseline'),
        'group_num': 2,
        'group_title': 'CIFAR-100 Baseline Comparison',
        'group_desc': 'Compares FedSALA against FedALA, FedAvg, and LocalOnly on CIFAR-100 with pathological non-IID splits.',
        'dataset': 'CIFAR-100',
        'model': 'ResNet-18',
        'rounds': 200,
        'lr': 0.001,
        'batch_size': 16,
        'extra_setup': 'FedSALA: Method 3, $\\tau=0.7527$, EMA$=0.5$, sample$=10\\%$. FedALA: layer\\_idx$=17$, $\\eta=1.0$, rand$=80\\%$.',
        'splits': CIFAR100_SPLITS,
        'split_desc': CIFAR100_SPLIT_DESC,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedALA',     'FedALA',     'FedALA'),
            ('FedAvg',     'FedAvg',     'FedAvg'),
            ('LocalOnly',  'LocalOnly',  'LocalOnly'),
        ],
    },
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group3_fedala_l_comparison'),
        'group_num': 3,
        'group_title': 'FedSALA vs FedALA-L (Layer-wise Lower-Local)',
        'group_desc': "Compares FedSALA (Fisher-informed selection) against FedALA-L (FedALA with \\texttt{--lower\\_layers\\_local}). FedALA-L keeps lower layers entirely local (never overwritten by global), mimicking FedSALA's freeze behavior but using a fixed layer boundary instead of Fisher-based selection.",
        'dataset': 'CIFAR-10',
        'model': 'ResNet-18',
        'rounds': 200,
        'lr': 0.001,
        'batch_size': 10,
        'extra_setup': 'FedSALA: Method 3, $\\tau=0.7516$, EMA$=0.5$, sample$=10\\%$. FedALA-L: layer\\_idx$=17$ + lower\\_layers\\_local.',
        'splits': CIFAR10_SPLITS,
        'split_desc': CIFAR10_SPLIT_DESC,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedALA-L',   'FedALA-L',   'FedALA'),
        ],
    },
    {
        'group_dir': os.path.join(FOUR_GROUPS_BASE, 'group4_fedsala_g_comparison'),
        'group_num': 4,
        'group_title': 'FedSALA vs FedSALA-G (Method Variant)',
        'group_desc': "Compares FedSALA Method 3 (High-Fisher$\\to$ALA, Low-Fisher$\\to$Freeze) against FedSALA-G Method 1 (High-Fisher$\\to$ALA, Low-Fisher$\\to$Global). Tests whether freezing low-importance parameters outperforms overwriting them with global weights.",
        'dataset': 'CIFAR-10',
        'model': 'ResNet-18',
        'rounds': 200,
        'lr': 0.001,
        'batch_size': 10,
        'extra_setup': 'FedSALA (M3): $\\tau=0.7516$, EMA$=0.5$, sample$=10\\%$. FedSALA-G (M1): same hyperparameters.',
        'splits': CIFAR10_SPLITS,
        'split_desc': CIFAR10_SPLIT_DESC,
        'methods': [
            ('FedSALA',    'FedSALA',    'FedSALA'),
            ('FedSALA-G',  'FedSALA-G',  'FedSALA'),
        ],
    },
    {
        'group_dir': os.path.join(RESULTS_BASE, 'fedsala_75_vs_100'),
        'group_num': 5,
        'group_title': 'FedSALA-75 vs FedSALA-100 (Threshold Ablation)',
        'group_desc': "Ablation study comparing the default Fisher threshold ($\\tau=0.7516$, 75.16\\% parameter coverage) against full coverage ($\\tau=1.0$, 100\\% all parameters in ALA zone). Tests whether Fisher-based parameter selection provides benefit over applying ALA to all parameters.",
        'dataset': 'CIFAR-10',
        'model': 'ResNet-18',
        'rounds': 200,
        'lr': 0.001,
        'batch_size': 10,
        'extra_setup': 'FedSALA-75: $\\tau=0.7516$, EMA$=0.5$, sample$=10\\%$. FedSALA-100: $\\tau=1.0$, EMA$=0.5$, sample$=10\\%$.',
        'splits': CIFAR10_SPLITS,
        'split_desc': CIFAR10_SPLIT_DESC,
        'methods': [
            ('FedSALA-75',  'FedSALA-75',  'FedSALA'),
            ('FedSALA-100', 'FedSALA-100', 'FedSALA'),
        ],
    },
]


def generate_latex():
    lines = []

    # Preamble
    lines.append(r'\documentclass[11pt,a4paper]{article}')
    lines.append(r'\usepackage[margin=1.8cm]{geometry}')
    lines.append(r'\usepackage{graphicx}')
    lines.append(r'\usepackage{booktabs}')
    lines.append(r'\usepackage{caption}')
    lines.append(r'\usepackage{subcaption}')
    lines.append(r'\usepackage{float}')
    lines.append(r'\usepackage{longtable}')
    lines.append(r'\usepackage{hyperref}')
    lines.append(r'\usepackage{amsmath}')
    lines.append(r'\usepackage[table]{xcolor}')
    lines.append(r'\usepackage{pdflscape}')
    lines.append(r'')
    lines.append(r'\definecolor{bestcolor}{RGB}{0,120,0}')
    lines.append(r'')
    lines.append(r'\title{FedSALA Experiment Results\\All Groups (1--5)}')
    lines.append(r'\author{FedSALA Research}')
    lines.append(r'\date{' + datetime.now().strftime('%Y-%m-%d') + '}')
    lines.append(r'')
    lines.append(r'\begin{document}')
    lines.append(r'\maketitle')
    lines.append(r'\tableofcontents')
    lines.append(r'\newpage')
    lines.append(r'')

    # Note about best accuracy
    lines.append(r'\noindent\textbf{Note:} Throughout this document, ``Best Accuracy" refers to the \textbf{highest individual metric value} observed across all 200 communication rounds. It is \textit{not} selected by the monitoring score (which is a combined average of skewed and whole-label accuracy). The best round for skewed accuracy and whole-label accuracy may differ.')
    lines.append(r'\vspace{1em}')
    lines.append(r'')

    for group in GROUPS:
        gnum = group['group_num']
        gdir = os.path.abspath(group['group_dir'])
        splits = group['splits']
        split_desc = group['split_desc']
        methods = group['methods']
        method_labels = [m[0] for m in methods]
        n_methods = len(methods)
        dir_basename = os.path.basename(group['group_dir'])

        lines.append(r'')
        lines.append(r'%% ================================================================')
        lines.append(f'%% GROUP {gnum}')
        lines.append(r'%% ================================================================')
        lines.append(f'\\section{{Group {gnum}: {group["group_title"]}}}')
        lines.append(r'')

        # --- Experiment Setup ---
        lines.append(f'\\subsection{{Experiment Setup}}')
        lines.append(f'{group["group_desc"]}')
        lines.append(r'')
        lines.append(r'\begin{itemize}')
        lines.append(f'  \\item \\textbf{{Dataset:}} {group["dataset"]}')
        lines.append(f'  \\item \\textbf{{Model:}} {group["model"]}')
        lines.append(f'  \\item \\textbf{{Communication Rounds:}} {group["rounds"]}')
        lines.append(f'  \\item \\textbf{{Learning Rate:}} {group["lr"]}')
        lines.append(f'  \\item \\textbf{{Batch Size:}} {group["batch_size"]}')
        lines.append(f'  \\item \\textbf{{Local Epochs:}} 1')
        lines.append(f'  \\item \\textbf{{Methods:}} {", ".join(method_labels)}')
        lines.append(f'  \\item {group["extra_setup"]}')
        lines.append(r'\end{itemize}')
        lines.append(r'')

        # --- Data Split Table ---
        lines.append(f'\\subsection{{Data Splits}}')
        lines.append(r'')
        lines.append(r'\begin{table}[H]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{Data split configurations for Group {gnum}.}}')
        lines.append(r'\small')
        lines.append(r'\begin{tabular}{llrl}')
        lines.append(r'\toprule')
        lines.append(r'\textbf{Split ID} & \textbf{Scenario} & \textbf{Clients} & \textbf{Description} \\')
        lines.append(r'\midrule')
        for split_id in splits:

            scenario, nc, desc = split_desc[split_id]
            sid_tex = split_id.replace('_', '\\_')
            lines.append(f'{sid_tex} & {scenario} & {nc} & {desc} \\\\')
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
        lines.append(r'')

        # --- Load all data ---
        all_data = {}  # all_data[split_id][method_label] = {sk_best, sk_rnd, sk_final, gl_best, gl_rnd, gl_final}
        for split_id in splits:
            all_data[split_id] = {}
            for label, folder_suffix, algo_prefix in methods:
                result_dir = os.path.join(gdir, f'{split_id}_{folder_suffix}')
                skewed = get_skewed_acc(result_dir, algo_prefix)
                global_acc = get_global_acc(result_dir, algo_prefix)
                sk_best, sk_rnd, sk_final = get_stats(skewed)
                gl_best, gl_rnd, gl_final = get_stats(global_acc)
                all_data[split_id][label] = {
                    'sk_best': sk_best, 'sk_rnd': sk_rnd, 'sk_final': sk_final,
                    'gl_best': gl_best, 'gl_rnd': gl_rnd, 'gl_final': gl_final,
                }

        # --- TABLE: Non-IID (Skewed) Test Accuracy ---
        lines.append(f'\\subsection{{Non-IID (Skewed) Test Accuracy}}')
        lines.append(r'')
        lines.append(r'\begin{table}[H]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{Non-IID (skewed label) test accuracy for Group {gnum}. Best Acc = max across all rounds; @Rnd = round at which it occurred; Final = accuracy at round 200. \\textbf{{Bold}} = best among methods.}}')
        lines.append(r'\small')

        # Column spec: Split | (Best, @Rnd, Final) per method
        col_spec = 'l' + '|rrr' * n_methods
        if n_methods >= 4:
            lines.append(r'\resizebox{\textwidth}{!}{')
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append(r'\toprule')

        # Method header row
        hdr = r' '
        for label in method_labels:
            label_tex = label.replace('_', '\\_')
            hdr += f' & \\multicolumn{{3}}{{c}}{{{label_tex}}}'
        hdr += r' \\'
        lines.append(hdr)

        # Sub-header
        sub = r'\textbf{Split}'
        for _ in method_labels:
            sub += r' & \textbf{Best} & \textbf{@Rnd} & \textbf{Final}'
        sub += r' \\'
        lines.append(r'\cmidrule(lr){1-1}' + ''.join([f'\\cmidrule(lr){{{2+3*i}-{4+3*i}}}' for i in range(n_methods)]))
        lines.append(sub)
        lines.append(r'\midrule')

        for split_id in splits:

            sid_tex = split_id.replace('_', '\\_')
            row = f'{sid_tex}'
            # Collect best and final values across methods for bolding
            best_vals = [all_data[split_id][m[0]]['sk_best'] for m in methods]
            final_vals = [all_data[split_id][m[0]]['sk_final'] for m in methods]
            valid_bests = [v for v in best_vals if v is not None]
            valid_finals = [v for v in final_vals if v is not None]
            max_best = max(valid_bests) if valid_bests else None
            max_final = max(valid_finals) if valid_finals else None

            for label, _, _ in methods:
                d = all_data[split_id][label]
                b = d['sk_best']
                f = d['sk_final']
                b_str = f'\\textbf{{{fmt(b)}}}' if (b is not None and max_best is not None and b == max_best) else fmt(b)
                f_str = f'\\textbf{{{fmt(f)}}}' if (f is not None and max_final is not None and f == max_final) else fmt(f)
                row += f' & {b_str} & {fmt_rnd(d["sk_rnd"])} & {f_str}'
            row += r' \\'
            lines.append(row)

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        if n_methods >= 4:
            lines.append(r'}')
        lines.append(r'\end{table}')
        lines.append(r'')

        # --- TABLE: Whole-Label (Global) Test Accuracy ---
        lines.append(f'\\subsection{{Whole-Label Test Accuracy}}')
        lines.append(r'')
        lines.append(r'\begin{table}[H]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{Whole-label (global) test accuracy for Group {gnum}. Best Acc = max across all rounds; @Rnd = round at which it occurred; Final = accuracy at round 200. \\textbf{{Bold}} = best among methods.}}')
        lines.append(r'\small')
        if n_methods >= 4:
            lines.append(r'\resizebox{\textwidth}{!}{')
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append(r'\toprule')
        lines.append(hdr)
        lines.append(r'\cmidrule(lr){1-1}' + ''.join([f'\\cmidrule(lr){{{2+3*i}-{4+3*i}}}' for i in range(n_methods)]))
        lines.append(sub)
        lines.append(r'\midrule')

        for split_id in splits:

            sid_tex = split_id.replace('_', '\\_')
            row = f'{sid_tex}'
            best_vals = [all_data[split_id][m[0]]['gl_best'] for m in methods]
            final_vals = [all_data[split_id][m[0]]['gl_final'] for m in methods]
            valid_bests = [v for v in best_vals if v is not None]
            valid_finals = [v for v in final_vals if v is not None]
            max_best = max(valid_bests) if valid_bests else None
            max_final = max(valid_finals) if valid_finals else None

            for label, _, _ in methods:
                d = all_data[split_id][label]
                b = d['gl_best']
                f = d['gl_final']
                b_str = f'\\textbf{{{fmt(b)}}}' if (b is not None and max_best is not None and b == max_best) else fmt(b)
                f_str = f'\\textbf{{{fmt(f)}}}' if (f is not None and max_final is not None and f == max_final) else fmt(f)
                row += f' & {b_str} & {fmt_rnd(d["gl_rnd"])} & {f_str}'
            row += r' \\'
            lines.append(row)

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        if n_methods >= 4:
            lines.append(r'}')
        lines.append(r'\end{table}')
        lines.append(r'')

        # --- FIGURES: Comparison Graphs ---
        lines.append(f'\\subsection{{Comparison Graphs}}')
        lines.append(r'')

        graph_splits = splits

        # Non-IID (Skewed) graphs first
        lines.append(f'\\subsubsection{{Non-IID (Skewed) Test Accuracy Graphs}}')
        lines.append(r'')

        # Show graphs in a 2-column layout, 2 per row
        for i in range(0, len(graph_splits), 2):
            lines.append(r'\begin{figure}[H]')
            lines.append(r'\centering')
            batch = graph_splits[i:i+2]
            for split_id in batch:
                fig_name = f'{dir_basename}_{split_id}_comparison_skewed.png'
                width = '0.48' if len(batch) == 2 else '0.6'
                lines.append(r'\begin{subfigure}{' + width + r'\textwidth}')
                lines.append(r'\centering')
                lines.append(f'\\includegraphics[width=\\textwidth]{{figures/{fig_name}}}')
                sid_tex = split_id.replace('_', '\\_')
                lines.append(f'\\caption{{{sid_tex}}}')
                lines.append(r'\end{subfigure}')
                if split_id != batch[-1]:
                    lines.append(r'\hfill')
            sid_range = ' \\& '.join([s.replace('_', '\\_') for s in batch])
            lines.append(f'\\caption{{Non-IID test accuracy: {sid_range} (Group {gnum})}}')
            lines.append(r'\end{figure}')
            lines.append(r'')

        # Whole-Label (Global) graphs
        lines.append(f'\\subsubsection{{Whole-Label Test Accuracy Graphs}}')
        lines.append(r'')

        for i in range(0, len(graph_splits), 2):
            lines.append(r'\begin{figure}[H]')
            lines.append(r'\centering')
            batch = graph_splits[i:i+2]
            for split_id in batch:
                fig_name = f'{dir_basename}_{split_id}_comparison_global.png'
                width = '0.48' if len(batch) == 2 else '0.6'
                lines.append(r'\begin{subfigure}{' + width + r'\textwidth}')
                lines.append(r'\centering')
                lines.append(f'\\includegraphics[width=\\textwidth]{{figures/{fig_name}}}')
                sid_tex = split_id.replace('_', '\\_')
                lines.append(f'\\caption{{{sid_tex}}}')
                lines.append(r'\end{subfigure}')
                if split_id != batch[-1]:
                    lines.append(r'\hfill')
            sid_range = ' \\& '.join([s.replace('_', '\\_') for s in batch])
            lines.append(f'\\caption{{Whole-label test accuracy: {sid_range} (Group {gnum})}}')
            lines.append(r'\end{figure}')
            lines.append(r'')

        lines.append(r'\newpage')
        lines.append(r'')

    lines.append(r'\end{document}')

    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  ✓ LaTeX file written to: {OUTPUT_FILE}")
    print(f"    Compile with: pdflatex experiment_results_all_groups.tex")


if __name__ == '__main__':
    generate_latex()
