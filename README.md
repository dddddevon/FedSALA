# FedSALA: Fisher-informed Selective Adaptive Local Aggregation for Personalized Federated Learning

FedSALA extends [FedALA](https://ojs.aaai.org/index.php/AAAI/article/view/26330) (AAAI 2023) by replacing its fixed layer-based parameter selection with **Fisher Information-based parameter-wise selection**. Instead of deciding which layers to personalize based on a manually-set index, FedSALA computes per-parameter Fisher Information values from each client's local data and uses a threshold to dynamically identify which parameters are most important for personalization.

## Key Idea

In FedALA, the boundary between "personalized" (ALA-blended) and "global" (directly overwritten) parameters is set at a **fixed layer index** — the same for all clients regardless of their data distribution. FedSALA replaces this with a **data-driven, per-parameter decision**:

1. **Compute Fisher Information** for every model parameter using each client's local data
2. **Apply EMA smoothing** across rounds to stabilize the Fisher mask
3. **Threshold**: parameters with cumulative Fisher value above τ (e.g., 75.16%) → **ALA blending** (personalized)
4. **Below threshold** → **Freeze as local** (Method 3, default) — preserving locally-learned features

This allows each client to have a different personalization mask that reflects its own data characteristics.

## Method Variants

| Method | High-Fisher Params | Low-Fisher Params | Description |
|:-------|:-------------------|:-------------------|:------------|
| **M1** (FedSALA-G) | ALA blending | Global overwrite | Less effective — overwrites unimportant but locally-tuned params |
| M2 | Freeze (local) | ALA blending | Inverse of M3 |
| **M3** (FedSALA) ★ | ALA blending | Freeze (local) | **Default** — best performance across all splits |
| M4 | Global overwrite | ALA blending | Inverse of M1 |

## Repository Structure

```
FedSALA_github/
├── README.md
├── LICENSE                                 # MIT License
├── requirements.txt                        # Python dependencies
├── .gitignore
│
├── system/
│   ├── main.py                             # Entry point — CLI args, training loop
│   ├── generate_cifar10.py                 # CIFAR-10 non-IID data partitioner
│   ├── generate_cifar100.py                # CIFAR-100 non-IID data partitioner
│   │
│   ├── flcore/
│   │   ├── clients/
│   │   │   ├── clientSALA.py               # FedSALA client
│   │   │   └── clientALA.py                # FedALA client (baseline)
│   │   ├── servers/
│   │   │   ├── serverSALA.py               # FedSALA server
│   │   │   └── serverALA.py                # FedALA/FedAvg server
│   │   └── trainmodel/
│   │       └── models.py                   # ResNet-18, CNN architectures
│   │
│   ├── utils/
│   │   ├── SALA.py                         # FedSALA core — Fisher computation & selective ALA
│   │   ├── ALA.py                          # Original FedALA module
│   │   └── data_utils.py                   # Dataset loading utilities
│   │
│   ├── scenarios/                          # CIFAR-10 data split configurations (10 JSONs)
│   ├── scenarios_cifar100/                 # CIFAR-100 data split configurations (3 JSONs)
│   │
│   ├── run_all_fedsala_experiments.sh      # Master experiment runner (92 runs, 4 groups)
│   ├── run_fedsala_75_vs_100.sh            # Threshold ablation study (20 runs)
│   ├── compare_split_results.py            # Per-split comparison plots
│   ├── generate_total_summary.py           # Cross-split summary tables
│   ├── generate_experiment_results_latex.py # LaTeX table/figure generator
│   ├── README_experiments.md               # Detailed experiment guide
│   └── env_linux.yaml                      # Conda environment specification
│
├── figs/                                   # Architectural diagrams
│   ├── ALA.jpg
│   ├── correction.png
│   └── illustrate.jpg
│
└── dataset/                                # Generated data goes here (gitignored)
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.8+ with CUDA support
- NVIDIA GPU (recommended for training)

### Installation

```bash
git clone https://github.com/dddddevon/FedSALA.git
cd FedSALA
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Data

Generate a CIFAR-10 non-IID data split from a scenario configuration:

```bash
cd system
python3 generate_cifar10.py --config scenarios/comp_2label.json
```

This creates per-client `.npz` files in `../dataset/Cifar10/`.

### 2. Run a Single Experiment

```bash
# FedSALA (Method 3 — default)
python3 main.py \
    -algo FedSALA \
    --fedsala_method 3 \
    --fisher_threshold 0.7516455 \
    --fisher_ema_alpha 0.5 \
    --fisher_sample_percent 10 \
    -m resnet -data Cifar10 -nb 10 -nc 10 \
    -gr 200 -lr 0.001 -lbs 10 -ls 1 \
    -dev cuda

# FedALA (baseline)
python3 main.py \
    -algo FedALA \
    -p 17 -s 80 -et 1.0 \
    -m resnet -data Cifar10 -nb 10 -nc 10 \
    -gr 200 -lr 0.001 -lbs 10 -ls 1 \
    -dev cuda

# FedAvg
python3 main.py \
    -algo FedAvg \
    -m resnet -data Cifar10 -nb 10 -nc 10 \
    -gr 200 -lr 0.001 -lbs 10 -ls 1 \
    -dev cuda
```

### 3. Run Full Experiment Suite

To reproduce all paper results (92 experiment runs across 4 groups):

```bash
cd system
chmod +x run_all_fedsala_experiments.sh
nohup ./run_all_fedsala_experiments.sh > ../results/fedsala_all_run.log 2>&1 &
tail -f ../results/fedsala_all_run.log
```

For the Fisher threshold ablation study (75.16% vs 100%):

```bash
chmod +x run_fedsala_75_vs_100.sh
nohup ./run_fedsala_75_vs_100.sh > ../results/fedsala_75_vs_100_run.log 2>&1 &
```

## Data Split Scenarios

### CIFAR-10 (10 scenarios)

| Split ID | Scenario | Clients | Description |
|:---------|:---------|:--------|:------------|
| S2 | comp_2label | 10 | 2 labels/client, equal ratio [1:1], 400 samples |
| S3 | comp_3label | 10 | 3 labels/client, equal ratio [1:1:1], 600 samples |
| S4 | comp_4label | 10 | 4 labels/client, equal ratio [1:1:1:1], 800 samples |
| SC | remote_sensing | 10 | 3 labels/client, skewed ratio [4:2:1], 560 samples |
| SD | hospital_uniform | 10 | 3 labels/client, skewed ratio [3:2:1], 600 samples |
| SE | hospital_mixed | 10 | 4 clients: 4 labels [4:3:2:1] 1600, 6 clients: 2 labels [3:1] 800 |
| SF | camera_trap | 15 | 2 labels/client, extreme skew [6:1], 1260 samples |
| SA2 | app_2label_skewed | 15 | 2 labels/client, skewed ratio [3:1], 400 samples |
| SA4 | app_4label_skewed | 5 | 4 labels/client, skewed ratio [4:3:2:1], 800 samples |
| SMX | app_mixed_hetero | 10 | 5 clients: 2 labels [3:1] 400, 5 clients: 4 labels [4:3:2:1] 800 |

### CIFAR-100 (3 scenarios)

| Split ID | Scenario | Clients | Description |
|:---------|:---------|:--------|:------------|
| C100_5c | fedala_pathological_5c | 20 | 5 classes/client, 500 samples/label (extreme) |
| C100_10c | fedala_pathological_10c | 20 | 10 classes/client, 250 samples/label (standard) |
| C100_20c | fedala_pathological_20c | 20 | 20 classes/client, 125 samples/label (mild) |

## Experiment Groups

The master experiment runner (`run_all_fedsala_experiments.sh`) executes 4 groups:

| Group | Comparison | Dataset | Runs |
|:------|:-----------|:--------|:-----|
| 1 | FedSALA vs FedALA vs FedAvg vs LocalOnly | CIFAR-10 | 40 |
| 2 | FedSALA vs FedALA vs FedAvg vs LocalOnly | CIFAR-100 | 12 |
| 3 | FedSALA vs FedALA-L (lower layers local) | CIFAR-10 | 20 |
| 4 | FedSALA vs FedSALA-G (Method 1 vs Method 3) | CIFAR-10 | 20 |

Additionally, `run_fedsala_75_vs_100.sh` runs a separate threshold ablation (20 runs).

## FedSALA-Specific CLI Flags

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--fedsala_method` | Method variant (1–4, see table above) | 3 |
| `--fisher_threshold` | Cumulative Fisher threshold τ (0–1) | 0.7516455 |
| `--fisher_ema_alpha` | EMA smoothing factor α (0 = no smoothing) | 0.5 |
| `--fisher_sample_percent` | % of local data used for Fisher computation | 10 |

## Hyperparameters

| Parameter | CIFAR-10 | CIFAR-100 |
|:----------|:---------|:----------|
| Fisher threshold τ | 0.7516455 (75.16%) | 0.7526674 (75.27%) |
| EMA α | 0.5 | 0.5 |
| Fisher sample % | 10 | 10 |
| FedALA layer_idx | 17 | 17 |
| Learning rate | 0.001 | 0.001 |
| Batch size | 10 | 16 |
| Communication rounds | 200 | 200 |
| Local epochs | 1 | 1 |
| Model | ResNet-18 | ResNet-18 |

> **Note:** The Fisher threshold is calibrated so that 75.16% (CIFAR-10) / 75.27% (CIFAR-100) of ResNet-18 parameters fall in the ALA zone, matching FedALA's `layer_idx=17` coverage exactly (0 parameter difference).

## Acknowledgments

This implementation is built upon the [FedALA](https://github.com/TsingZ0/FedALA) codebase (AAAI 2023). We gratefully acknowledge the original authors:

```bibtex
@inproceedings{zhang2023fedala,
  title={FedALA: Adaptive Local Aggregation for Personalized Federated Learning},
  author={Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={9},
  pages={11237--11244},
  year={2023}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
