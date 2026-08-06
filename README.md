# FedSALA: Selective Adaptive Local Aggregation for Personalized Federated Learning

FedSALA is a personalized federated learning method that uses **Fisher Information** as a data-driven criterion to identify prediction-sensitive parameters across all model layers. Unlike layer-based selection approaches that rely on architectural heuristics, FedSALA computes per-parameter Fisher values from each client's local data and dynamically determines which parameters should receive adaptive local-global blending (ALA) and which should preserve their locally-learned values.

## Overview

In personalized federated learning, clients need to integrate shared global knowledge with locally-trained parameters. Existing methods either select parameters based on **layer position** (an architectural heuristic) or use **update magnitude** (which measures optimization movement rather than prediction sensitivity).

FedSALA addresses this with a **Fisher-based Selective Adaptive Local Aggregation (SALA)** pipeline:

1. **Compute Diagonal Fisher Information** — estimate how sensitive each parameter is to the client's local predictions
2. **Apply EMA Smoothing** — stabilize Fisher estimates across communication rounds using exponential moving average
3. **Generate Binary Mask** — threshold the top P% of parameters by Fisher value (default: 75.16%)
4. **High-Fisher parameters (ALA zone)** → receive adaptive local-global blending with learned aggregation weights
5. **Low-Fisher parameters (Local zone)** → preserve locally-learned values, keeping global knowledge out

This produces a **per-client, per-parameter** personalization mask that reflects each client's own data characteristics — capturing prediction-critical weights wherever they occur in the model, not just in upper layers.

### FedSALA-G Variant

FedSALA-G differs from FedSALA in how it treats low-Fisher parameters:

| Variant | High-Fisher Params | Low-Fisher Params |
|:--------|:-------------------|:-------------------|
| **FedSALA** (default) | ALA blending | Preserve local values |
| **FedSALA-G** | ALA blending | Global overwrite |

Experiments (Section IV-E of the paper) show that preserving local values for low-Fisher parameters (FedSALA) consistently outperforms global overwriting (FedSALA-G), validating that low-Fisher parameters should retain locally-learned knowledge.

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
│   │       └── models.py                   # ResNet-18 architecture
│   │
│   ├── utils/
│   │   ├── SALA.py                         # FedSALA core — Fisher computation & selective ALA
│   │   ├── ALA.py                          # ALA module (baseline)
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
# FedSALA (default)
python3 main.py \
    -algo FedSALA \
    --fedsala_method 3 \
    --fisher_threshold 0.7516455 \
    --fisher_ema_alpha 0.5 \
    --fisher_sample_percent 10 \
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

## Experiment Groups

The master experiment runner (`run_all_fedsala_experiments.sh`) executes 4 groups:

| Group | Comparison | Dataset | Runs | Paper Section |
|:------|:-----------|:--------|:-----|:--------------|
| 1 | FedSALA vs FedALA vs FedAvg vs Local-only | CIFAR-10 | 40 | IV-B |
| 2 | FedSALA vs FedALA vs FedAvg vs Local-only | CIFAR-100 | 12 | IV-C |
| 3 | FedSALA vs FedALA-L (parameter-wise vs layer-wise selection) | CIFAR-10 | 20 | IV-D |
| 4 | FedSALA vs FedSALA-G (local preservation vs global overwrite) | CIFAR-10 | 20 | IV-E |

Additionally, `run_fedsala_75_vs_100.sh` runs a separate threshold ablation comparing selective (75.16%) vs full (100%) ALA zone coverage (Section IV-F).

## Data Split Scenarios

### CIFAR-10 (10 scenarios)

| Split ID | Clients | Description |
|:---------|:--------|:------------|
| S2 | 10 | 2 labels/client, equal ratio [1:1], 400 samples |
| S3 | 10 | 3 labels/client, equal ratio [1:1:1], 600 samples |
| S4 | 10 | 4 labels/client, equal ratio [1:1:1:1], 800 samples |
| SC | 10 | 3 labels/client, skewed ratio [4:2:1], 560 samples |
| SD | 10 | 3 labels/client, skewed ratio [3:2:1], 600 samples |
| SE | 10 | 4 clients: 4 labels [4:3:2:1] 1600, 6 clients: 2 labels [3:1] 800 |
| SF | 15 | 2 labels/client, extreme skew [6:1], 1260 samples |
| SA2 | 15 | 2 labels/client, skewed ratio [3:1], 400 samples |
| SA4 | 5 | 4 labels/client, skewed ratio [4:3:2:1], 800 samples |
| SMX | 10 | 5 clients: 2 labels [3:1] 400, 5 clients: 4 labels [4:3:2:1] 800 |

### CIFAR-100 (3 scenarios)

| Split ID | Clients | Description |
|:---------|:--------|:------------|
| C100_5c | 20 | 5 classes/client, 500 samples/label (extreme) |
| C100_10c | 20 | 10 classes/client, 250 samples/label (standard) |
| C100_20c | 20 | 20 classes/client, 125 samples/label (mild) |

## FedSALA-Specific CLI Flags

| Flag | Description | Default |
|:-----|:------------|:--------|
| `--fedsala_method` | Method variant (3 = FedSALA, 1 = FedSALA-G) | 3 |
| `--fisher_threshold` | Cumulative Fisher threshold τ (0–1) | 0.7516455 |
| `--fisher_ema_alpha` | EMA smoothing factor α (0 = no smoothing) | 0.5 |
| `--fisher_sample_percent` | % of local data used for Fisher computation | 10 |

## Training Settings

| Parameter | CIFAR-10 | CIFAR-100 |
|:----------|:---------|:----------|
| Fisher threshold τ | 0.7516455 (75.16%) | 0.7526674 (75.27%) |
| EMA α | 0.5 | 0.5 |
| Fisher sample % | 10 | 10 |
| Learning rate | 0.001 | 0.001 |
| Batch size | 10 | 16 |
| Communication rounds | 200 | 200 |
| Local epochs | 1 | 1 |
| Model | ResNet-18 | ResNet-18 |
| Optimizer | SGD (momentum=0.9, weight decay=5e-4) | SGD (momentum=0.9, weight decay=5e-4) |
| LR schedule | Cosine annealing | Cosine annealing |
| Client participation | 100% | 100% |

> **Note:** The Fisher threshold is calibrated so that 75.16% (CIFAR-10) / 75.27% (CIFAR-100) of ResNet-18 parameters are placed in the ALA zone.

## Acknowledgment

This implementation is built upon the [FedALA](https://github.com/TsingZ0/FedALA) codebase (AAAI 2023). We gratefully acknowledge the original authors for making their code publicly available.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
