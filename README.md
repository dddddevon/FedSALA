# FedSALA: Selective Adaptive Local Aggregation for Personalized Federated Learning

This is the implementation of *FedSALA: Selective Adaptive Local Aggregation for Personalized Federated Learning*.

FedSALA replaces layer-position-based parameter selection with **Fisher Information-based parameter-wise selection**, enabling each client to adaptively identify prediction-sensitive parameters across all layers for personalized local-global aggregation.

## Requirements

The code requires Python 3.8+ and the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

CUDA-enabled GPU is recommended for training.

## Datasets

Data is generated using JSON-based scenario configurations. The `scenarios/` and `scenarios_cifar100/` directories contain predefined non-IID split configurations for CIFAR-10 (10 scenarios) and CIFAR-100 (3 scenarios).

```bash
cd system
python3 generate_cifar10.py --config scenarios/comp_2label.json
python3 generate_cifar100.py --config scenarios_cifar100/fedala_pathological_5c.json
```

## System

- `main.py`: configurations and entry point for **FedSALA**.
- `run_all_fedsala_experiments.sh`: run all 4 experiment groups (92 runs).
- `run_fedsala_75_vs_100.sh`: threshold ablation study (20 runs).
- `./flcore`:
    - `./clients/clientSALA.py`: the code on the client.
    - `./clients/clientALA.py`: the baseline client (FedALA).
    - `./servers/serverSALA.py`: the code on the server.
    - `./servers/serverALA.py`: the baseline server.
    - `./trainmodel/models.py`: the code for backbones.
- `./utils`:
    - `SALA.py`: the code of our **Selective Adaptive Local Aggregation (SALA)** module.
    - `ALA.py`: the baseline ALA module.
    - `data_utils.py`: the code to read the dataset.
- `./scenarios/`: CIFAR-10 non-IID split configurations (10 JSONs).
- `./scenarios_cifar100/`: CIFAR-100 non-IID split configurations (3 JSONs).
- `compare_split_results.py`: per-split comparison plot generator.
- `generate_total_summary.py`: cross-split summary table generator.

## SALA Module

`./system/utils/SALA.py` is the implementation of the SALA module. It performs the following steps each communication round (from round 2 onward):

1. **Fisher Computation**: Estimate diagonal Fisher Information from a sampled subset of local data.
2. **EMA Smoothing**: Smooth Fisher values across rounds to stabilize the binary mask.
3. **Mask Generation**: Threshold the top P% parameters by Fisher value → binary mask M.
4. **Initialization**: High-Fisher parameters (M=1) are initialized with a weighted blend of local and global values. Low-Fisher parameters (M=0) retain their local values.
5. **Weight Learning**: Learn aggregation weights for M=1 parameters via gradient descent on local data, gated by the mask.

### How to use

```bash
cd system

# FedSALA
python3 main.py -algo FedSALA --fedsala_method 3 \
    --fisher_threshold 0.7516455 --fisher_ema_alpha 0.5 \
    --fisher_sample_percent 10 \
    -m resnet -data Cifar10 -nb 10 -nc 10 \
    -gr 200 -lr 0.001 -lbs 10 -ls 1 -dev cuda

# FedALA (baseline)
python3 main.py -algo FedALA -p 17 -s 80 -et 1.0 \
    -m resnet -data Cifar10 -nb 10 -nc 10 \
    -gr 200 -lr 0.001 -lbs 10 -ls 1 -dev cuda

# Run all experiments
chmod +x run_all_fedsala_experiments.sh
nohup ./run_all_fedsala_experiments.sh > ../results/run.log 2>&1 &
```

## Acknowledgment

This implementation is built upon the [FedALA](https://github.com/TsingZ0/FedALA) codebase (AAAI 2023). We gratefully acknowledge the original authors for making their code publicly available.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
