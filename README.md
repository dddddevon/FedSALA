# FedSALA: Selective Adaptive Local Aggregation for Personalized Federated Learning

This is the implementation of *FedSALA: Selective Adaptive Local Aggregation for Personalized Federated Learning*.

FedSALA enable each client to adaptively identify prediction-sensitive parameters across all layers for local-global aggregation.

## Requirements

The code requires Python 3.8+ and the dependencies listed in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA-enabled GPU is recommended for training.

## Datasets

Data is generated using JSON-based scenario configurations:

```bash
cd system
python3 generate_cifar10.py --config scenarios/comp_2label.json
python3 generate_cifar100.py --config scenarios_cifar100/fedala_pathological_5c.json
```

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

## How to Use

All codes corresponding to **FedSALA** are stored in `./system`. 

To reproduce all paper results (4 experiment groups, 92 runs total):

```bash
cd system
chmod +x run_all_fedsala_experiments.sh
nohup ./run_all_fedsala_experiments.sh > ../results/run.log 2>&1 &
```

To run the threshold ablation study (75.16% vs 100%, 20 runs):

```bash
chmod +x run_fedsala_75_vs_100.sh
nohup ./run_fedsala_75_vs_100.sh > ../results/run_75_vs_100.log 2>&1 &
```

## Acknowledgment

This implementation is built upon the [FedALA](https://github.com/TsingZ0/FedALA) codebase (AAAI 2023). We gratefully acknowledge the original authors for making their code publicly available.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
