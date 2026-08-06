# FedSALA Experiment Guide
## How to Run All 5 Baselines

---

### Quick Start

All experiments are run from the `FedALA/system/` directory:

```bash
cd /home/devon/Desktop/fedALA_practice/FedALA/system
```

The entry point is **`main.py`**. You just pass different `-algo` flags
to switch between algorithms. Everything else stays the same.

---

### Prerequisites

Before running, you need:

1. **CIFAR-10 dataset generated as `.npz` files** in `../dataset/Cifar10/`
   - Each client gets its own train/test npz file
   - The data partitioning script creates pathological non-IID splits (K classes per client)

2. **PyTorch, NumPy, scikit-learn** installed in your Python environment

---

## The 5 Baselines

### 1. FedSALA (Our Method)

Fisher-based parameter selection + ALA weight learning.
Parameters with high Fisher values get personalized (ALA blend),
the rest just take the global value.

```bash
python3 main.py \
    -algo FedSALA \
    -m resnet \
    -data Cifar10 \
    -nb 10 \
    -nc 20 \
    -gr 200 \
    -ls 5 \
    -lr 0.01 \
    -lbs 128 \
    -et 1.0 \
    -s 80 \
    --fisher_threshold 0.5 \
    --fisher_ema_alpha 0.9 \
    --fisher_sample_percent 10 \
    -dev cuda
```

**FedSALA-specific flags:**
| Flag | What it does | Recommended |
|:---|:---|:---|
| `--fisher_threshold 0.5` | Top 50% of params get ALA treatment | 0.3–0.7 |
| `--fisher_ema_alpha 0.9` | EMA smoothing (higher = more stable mask) | 0.8–0.95 |
| `--fisher_sample_percent 10` | Use 10% of local data for Fisher | 5–20 |


### 2. FedALA (Main Baseline to Beat)

Layer-based parameter selection + ALA weight learning.
The last N layers get personalized, the rest take global.

```bash
python3 main.py \
    -algo FedALA \
    -m resnet \
    -data Cifar10 \
    -nb 10 \
    -nc 20 \
    -gr 200 \
    -ls 5 \
    -lr 0.01 \
    -lbs 128 \
    -et 1.0 \
    -s 80 \
    -p 2 \
    -dev cuda
```

**Key flag:** `-p 2` means the last 2 layers (fc + last residual block) get ALA treatment.


### 3. FedAvg (Pure Averaging Baseline)

No personalization at all. Server just averages all client models.
This is the standard federated learning baseline.

```bash
python3 main.py \
    -algo FedAvg \
    -m resnet \
    -data Cifar10 \
    -nb 10 \
    -nc 20 \
    -gr 200 \
    -ls 5 \
    -lr 0.01 \
    -lbs 128 \
    -dev cuda
```

No extra flags needed — internally sets `layer_idx=0` (no ALA).


### 4. LocalOnly (No Communication Baseline)

Each client trains completely independently. No server aggregation at all.
This is the "lower bound" — shows how well clients do without sharing.

```bash
python3 main.py \
    -algo LocalOnly \
    -m resnet \
    -data Cifar10 \
    -nb 10 \
    -nc 20 \
    -gr 1 \
    -ls 50 \
    -lr 0.01 \
    -lbs 128 \
    -dev cuda
```

**Note:** `-gr 1` with `-ls 50` means each client trains for 50 local epochs
with no communication. Adjust `-ls` to match total computation budget.


### 5. Centralized (Upper Bound)

All data pooled into one client. Single model trained on everything.
This is the "upper bound" — the best you could do if privacy didn't matter.

```bash
python3 main.py \
    -algo Centralized \
    -m resnet \
    -data Cifar10 \
    -nb 10 \
    -nc 1 \
    -gr 200 \
    -ls 5 \
    -lr 0.01 \
    -lbs 128 \
    -dev cuda
```

**Note:** Requires a special dataset directory where all data goes to client 0.

---

## Common Flags Reference

| Flag | Long Name | What it does | Default |
|:---|:---|:---|:---|
| `-algo` | `--algorithm` | Which algorithm to run | FedALA |
| `-m` | `--model` | Model architecture (cnn, resnet, fastText) | cnn |
| `-data` | `--dataset` | Dataset folder name | mnist |
| `-nb` | `--num_classes` | Number of classes | 10 |
| `-nc` | `--num_clients` | Total number of clients | 20 |
| `-gr` | `--global_rounds` | Communication rounds | 1000 |
| `-ls` | `--local_steps` | Local training epochs per round | 1 |
| `-lr` | `--local_learning_rate` | Client-side learning rate | 0.005 |
| `-lbs` | `--batch_size` | Training batch size | 10 |
| `-jr` | `--join_ratio` | Fraction of clients per round | 1.0 |
| `-et` | `--eta` | ALA weight learning rate | 1.0 |
| `-s` | `--rand_percent` | % of data for ALA weight learning | 80 |
| `-p` | `--layer_idx` | FedALA: top N layers to personalize | 2 |
| `-dev` | `--device` | cuda or cpu | cuda |
| `-did` | `--device_id` | GPU device ID | 0 |
| `-eg` | `--eval_gap` | Evaluate every N rounds | 1 |
| `-t` | `--times` | Repeat experiment N times | 1 |

---

## Run All 5 Baselines (Batch Script)

Copy this shell script to run all experiments back-to-back:

```bash
#!/bin/bash
# run_all_experiments.sh
# Run from: FedALA/system/

COMMON="-m resnet -data Cifar10 -nb 10 -nc 20 -gr 200 -ls 5 -lr 0.01 -lbs 128 -dev cuda"

echo "===== 1/5: FedSALA ====="
python3 main.py -algo FedSALA $COMMON \
    --fisher_threshold 0.5 --fisher_ema_alpha 0.9 --fisher_sample_percent 10 \
    -et 1.0 -s 80

echo "===== 2/5: FedALA ====="
python3 main.py -algo FedALA $COMMON \
    -et 1.0 -s 80 -p 2

echo "===== 3/5: FedAvg ====="
python3 main.py -algo FedAvg $COMMON

echo "===== 4/5: LocalOnly ====="
python3 main.py -algo LocalOnly $COMMON -gr 1 -ls 50

echo "===== 5/5: Centralized ====="
python3 main.py -algo Centralized -m resnet -data Cifar10 -nb 10 \
    -nc 1 -gr 200 -ls 5 -lr 0.01 -lbs 128 -dev cuda

echo "All experiments complete!"
```

---

## Quick Smoke Test (3 rounds, 2 clients)

To verify everything works before running full experiments:

```bash
# Quick test with existing MNIST data
python3 main.py -algo FedSALA -m cnn -data mnist-0.1-npz -nb 10 \
    -nc 2 -gr 3 -ls 1 -lr 0.005 -lbs 10 -dev cpu \
    --fisher_threshold 0.5 --fisher_ema_alpha 0.9
```

If this runs without errors, you're good to go!

---

## File Structure

```
FedALA/system/
├── main.py                          ← Entry point (run this)
├── utils/
│   ├── ALA.py                       ← Original FedALA module
│   ├── SALA.py                      ← FedSALA module (Fisher + masked ALA)
│   └── data_utils.py                ← Data loading utilities
├── flcore/
│   ├── clients/
│   │   ├── clientALA.py             ← FedALA client
│   │   └── clientSALA.py            ← FedSALA client
│   ├── servers/
│   │   ├── serverALA.py             ← FedALA server (also used by FedAvg)
│   │   └── serverSALA.py            ← FedSALA server
│   └── trainmodel/
│       └── models.py                ← Model definitions (CNN, etc.)
└── ../dataset/
    └── Cifar10/ or mnist-0.1-npz/   ← Pre-generated data files
        ├── train/
        │   ├── 0.npz                ← Client 0's training data
        │   ├── 1.npz                ← Client 1's training data
        │   └── ...
        └── test/
            ├── 0.npz
            └── ...
```
