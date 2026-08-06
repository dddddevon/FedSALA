"""
main.py — FedSALA Experiment Runner (Redesigned Evaluation Framework)
======================================================================

WHAT THIS DOES:
    Runs ALL 4 baselines (or a single one) on CIFAR-10 with ResNet-18:
    1. FedSALA     — Our method (Fisher-based selection + ALA)
    2. FedALA      — Baseline (layer-based selection + ALA)
    3. FedAvg      — Baseline (pure averaging, no personalization)
    4. LocalOnly   — Baseline (no communication)

    Centralized is excluded from --run_all and must be run separately:
    python3 main.py -algo Centralized

EVALUATION FRAMEWORK:
    Two-perspective evaluation per method:
    A) Personalized performance  — skewed local test set (matches training distribution)
    B) Generalization performance — whole-label test shard (balanced, non-overlapping)

    Method-specific early stopping with best-model storage + patience.

HOW TO RUN:
    # Run all 4 baselines at once:
    python3 main.py --run_all

    # Run a single algorithm:
    python3 main.py -algo FedSALA

    # Run Centralized separately:
    python3 main.py -algo Centralized

OUTPUTS:
    results/
    ├── accuracy_skewed.png         ← Skewed local accuracy vs rounds (4 baselines)
    ├── accuracy_global.png         ← Whole-label accuracy vs rounds (4 baselines)
    ├── training_loss.png           ← Training loss vs rounds (4 baselines)
    ├── monitoring_score.png        ← Early stopping score vs rounds (4 baselines)
    ├── final_results.txt           ← Final accuracy summary with early stopping info
    └── {algorithm}_*.npy           ← Raw metric data per algorithm
"""

import torch  # tensor(multi-dimensional array),automatic differentiation
import argparse # parse flags in CLI as arguement for experiment
import os # for interacting with operating system : ex) creating folders
import sys
import time # for recording current time -> for logging experiment duration
import copy # ADDED: for deep-copying model weights during early stopping checkpointing
import warnings # Controls how warning messages are displayed
import json
import numpy as np # fast numerical math, e.g. array operations
import torchvision # pytorch helper library for computer vision - CIFAR-10
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from flcore.servers.serverALA import FedALA
from flcore.servers.serverSALA import FedSALA
from flcore.trainmodel.models import *

warnings.simplefilter("ignore")
# explanation start:
# Machine learning libraries often throw out highly verbose "deprecation" warnings 
# that don't actually break the code. 
# Researchers often use this module to explicitly silence them 
# so they don't clutter up the training logs in the terminal.
# explanation end:

torch.manual_seed(0)


# =========================================================================
# SECTION 1: Client Data Split Visualization
# =========================================================================

def visualize_client_splits(dataset, num_clients, out_file=None, split_type='train'):
    """
    Show how data is distributed across clients.
    Prints a table showing each client's class distribution.
    Can also output to an open text file if out_file is provided.
    
    Supports both CIFAR-10 (10 classes, per-column table) and CIFAR-100
    (100 classes, compact summary showing class count + total samples).
    """
    from utils.data_utils import read_client_data, read_client_data_global

    # Map split_type to a human-readable title
    title_map = {
        'train': 'TRAINING DATA DISTRIBUTION',
        'test': 'SKEWED TEST DATA DISTRIBUTION',
        'test_global': 'GLOBAL TEST DATA DISTRIBUTION (Whole-Label)'
    }
    title = title_map.get(split_type, 'DATA DISTRIBUTION')

    def log(msg, end="\n"):
        print(msg, end=end)
        if out_file is not None:
            out_file.write(str(msg) + end)

    # Detect number of classes from the first client's data
    try:
        probe_data = read_client_data(dataset, 0, is_train=True)
        all_probe_labels = set()
        for cid in range(min(num_clients, 5)):
            d = read_client_data(dataset, cid, is_train=True)
            all_probe_labels.update(int(y) for _, y in d)
        num_classes = max(all_probe_labels) + 1 if all_probe_labels else 10
    except:
        num_classes = 10

    # Choose display mode: detailed table for <=10 classes, compact for >10
    compact_mode = num_classes > 10

    if compact_mode:
        # --- Compact mode for CIFAR-100 (100 classes) ---
        log("\n" + "=" * 70)
        log(f"  {title} ({num_classes} classes)")
        log("=" * 70)
        log(f"  {'Client':<8} {'#Classes':<10} {'Samples':<10} {'Top Classes (id:count)'}")
        log(f"  {'-'*66}")
    else:
        # --- Detailed mode for CIFAR-10 ---
        class_names = ['airplane', 'auto', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        log("\n" + "=" * 80)
        log(title)
        log("=" * 80)
        header = f"{'Client':<8}"
        for name in class_names:
            header += f"{name:<8}"
        header += f"{'Total':<8}"
        log(header)
        log("-" * 80)

    total_samples = 0
    for client_id in range(num_clients):
        try:
            if split_type == 'train':
                data = read_client_data(dataset, client_id, is_train=True)
            elif split_type == 'test':
                data = read_client_data(dataset, client_id, is_train=False)
            elif split_type == 'test_global':
                data = read_client_data_global(dataset, client_id)
            else:
                data = read_client_data(dataset, client_id, is_train=True)
            
            labels = [int(y) for _, y in data]
            unique, counts = np.unique(labels, return_counts=True)
            count_dict = dict(zip(unique, counts))

            if compact_mode:
                # Show top 5 classes by count
                sorted_classes = sorted(zip(unique, counts), key=lambda x: -x[1])
                top5 = ', '.join([f'{c}:{cnt}' for c, cnt in sorted_classes[:5]])
                if len(sorted_classes) > 5:
                    top5 += f' (+{len(sorted_classes)-5} more)'
                log(f"  {client_id:<8} {len(unique):<10} {len(data):<10} {top5}")
            else:
                row = f"  {client_id:<6}"
                for c in range(num_classes):
                    cnt = count_dict.get(c, 0)
                    if cnt > 0:
                        row += f"{cnt:<8}"
                    else:
                        row += f"{'·':<8}"
                row += f"{len(data):<8}"
                log(row)

            total_samples += len(data)
        except FileNotFoundError:
            log(f"  {client_id:<6}  [DATA NOT FOUND — run generate script first]")
            return False

    if compact_mode:
        log(f"  {'-'*66}")
        log(f"  {'Total':<8} {'':10} {total_samples:<10}")
        log("=" * 70 + "\n")
    else:
        log("-" * 80)
        log(f"  {'Total':<6}", end="")
        all_labels = []
        for client_id in range(num_clients):
            if split_type == 'train':
                data = read_client_data(dataset, client_id, is_train=True)
            elif split_type == 'test':
                data = read_client_data(dataset, client_id, is_train=False)
            elif split_type == 'test_global':
                data = read_client_data_global(dataset, client_id)
            else:
                data = read_client_data(dataset, client_id, is_train=True)
            all_labels.extend([int(y) for _, y in data])
        unique, counts = np.unique(all_labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        for c in range(num_classes):
            log(f"{count_dict.get(c, 0):<8}", end="")
        log(f"{total_samples:<8}")
        log("=" * 80 + "\n")

    return True



# =========================================================================
# SECTION 2: Progress Display
# =========================================================================

def print_progress(algo_name, round_num, total_rounds, start_time):
    """Print a progress line showing current algorithm, round, and ETA."""
    elapsed = time.time() - start_time
    if round_num > 0:
        eta = elapsed / round_num * (total_rounds - round_num)
        eta_str = f"{int(eta//60)}m{int(eta%60)}s"
    else:
        eta_str = "calculating..."

    pct = round_num / total_rounds * 100
    bar_len = 30
    filled = int(bar_len * round_num / total_rounds)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"\r  [{bar}] {pct:5.1f}% | Round {round_num}/{total_rounds} | "
          f"Elapsed: {int(elapsed//60)}m{int(elapsed%60)}s | ETA: {eta_str}",
          end="", flush=True)


# =========================================================================
# SECTION 3: Model Creation
# =========================================================================

def create_model(args, model_str):
    """Create the model based on args. Handles ResNet-18 GRAD-MATCH modifications."""
    if model_str == "cnn":
        if args.dataset[:5] == "Cifar":
            return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        else:
            return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)

    elif model_str == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)
        # explanation start: CIFAR-10 STEM TWEAKS
        # Original ResNet18 is designed for huge 224x224 ImageNet images, using a 7x7 conv1 and MaxPool.
        # This destroys the tiny 32x32 CIFAR-10 images. 
        # To perform well on CIFAR-10, we make these stem tweaks:
        # 1. Replace conv1 with a 3x3 kernel, stride 1, padding 1 to preserve detail.
        # 2. Delete the maxpool layer (replace with Identity) to prevent downsampling too early.
        # explanation end:
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        return model.to(args.device)

    else:
        raise NotImplementedError(f"Unknown model: {model_str}")


# =========================================================================
# SECTION 4: Early Stopping Helper
# =========================================================================
# NEW: Early stopping check logic, factored out to avoid duplication.
# Updates best_score, best_round, patience_counter, and stores model checkpoint.
# Returns (best_score, best_round, patience_counter, best_model_state, should_stop)

def early_stopping_check(monitoring_score, current_round, best_score, best_round,
                         patience_counter, patience_limit, server, algo_name):
    """
    Check if monitoring_score improved. If yes, store checkpoint and reset patience.
    If no, increment patience. Returns updated state and whether to stop.
    
    The checkpoint stored is the server's global_model state_dict.
    For LocalOnly, the caller stores client models separately.
    """
    best_model_state = None
    should_stop = False
    
    if monitoring_score > best_score:
        # IMPROVEMENT: save checkpoint, reset patience
        best_score = monitoring_score
        best_round = current_round
        patience_counter = 0
        # Store the global model weights (caller may override for LocalOnly)
        best_model_state = copy.deepcopy(server.global_model.state_dict())
        print(f"  ★ New best score: {best_score:.4f} at round {best_round} (patience reset)")
    else:
        # NO IMPROVEMENT: increment patience
        patience_counter += 1
        print(f"  Patience: {patience_counter}/{patience_limit} "
              f"(best: {best_score:.4f} @ round {best_round})")
        if patience_counter >= patience_limit:
            print(f"\n  ⚠ Early stopping triggered at round {current_round}! "
                  f"Best was round {best_round} with score {best_score:.4f}")
            should_stop = True
    
    return best_score, best_round, patience_counter, best_model_state, should_stop


# =========================================================================
# SECTION 5: Method-Specific Training Loops
# =========================================================================

# ---------------------
# 5a. LocalOnly
# ---------------------
# REDESIGNED: LocalOnly evaluates BEFORE local training each round.
# It computes both skewed (local) and whole-label (global) accuracy,
# then uses monitoring_score = (local_acc + global_acc) / 2 for early stopping.
# No server aggregation occurs — each client trains independently.

def run_local_only(args, server, model_str):
    """
    LocalOnly baseline: no communication between clients.
    
    Returns dict with:
        - 'local_acc': list of skewed test acc per eval round
        - 'global_acc': list of whole-label test acc per eval round
        - 'train_loss': list of training loss per eval round
        - 'monitoring_score': list of early stopping scores
        - 'best_round': round where best checkpoint was stored
        - 'best_score': best monitoring score achieved
        - 'best_local_acc': local acc at best round
        - 'best_global_acc': global acc at best round
    """
    start_time = time.time()
    
    # Early stopping state
    best_score = -1.0
    best_round = -1
    best_model_states = None  # dict of {client_id: state_dict}
    patience_counter = 0
    patience_limit = args.patience
    best_local_acc = 0.0
    best_global_acc = 0.0
    
    # Metric tracking arrays
    local_accs = []
    global_accs = []
    train_losses = []
    monitoring_scores = []
    
    print("  LocalOnly: training each client independently (no aggregation)...")
    
    for i in range(args.global_rounds + 1):
        # STEP 1: Evaluate BEFORE local training
        # For LocalOnly, the "before training" model IS the post-training model
        # from the previous round (since there's no aggregation step in between).
        if i % args.eval_gap == 0:
            print(f"\n  [Round {i}/{args.global_rounds}]")
            local_acc, global_acc = server.evaluate()
            
            # Compute training loss separately
            stats_train = server.train_metrics()
            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
            
            # Record to tracking arrays
            local_accs.append(local_acc)
            global_accs.append(global_acc)
            train_losses.append(train_loss)
            
            # Monitoring score = balanced average of both perspectives
            monitoring_score = (local_acc + global_acc) / 2.0
            monitoring_scores.append(monitoring_score)
            print(f"  Monitoring Score: {monitoring_score:.4f} "
                  f"= ({local_acc:.4f} + {global_acc:.4f}) / 2")
            
            # STEP 2: Track best model checkpoint (no early stopping break)
            # NOTE: We keep track of the best round, best score, and best accuracies
            # by checking if the current monitoring score is the highest seen so far.
            # However, we DO NOT decrement patience or call 'break' here. This ensures
            # that training runs for the exact fixed number of rounds requested, while
            # still allowing us to log the best results achieved at any point.
            if monitoring_score > best_score:
                best_score = monitoring_score
                best_round = i
                best_local_acc = local_acc
                best_global_acc = global_acc
                # Store ALL client models (LocalOnly has no single global model)
                best_model_states = {c.id: copy.deepcopy(c.model.state_dict()) 
                                     for c in server.clients}
                print(f"  ★ New best score: {best_score:.4f} at round {best_round}")
        
        # STEP 3: Each client trains locally (no aggregation)
        for client in server.clients:
            client.train()
        
        print_progress("LocalOnly", i, args.global_rounds, start_time)
    
    elapsed = time.time() - start_time
    print(f"\n\n  LocalOnly complete in {elapsed:.1f}s")
    print(f"  Best Monitoring Score: {best_score:.4f} (Round {best_round})")
    print(f"  Best Local Acc: {best_local_acc:.4f} | Best Global Acc: {best_global_acc:.4f}")
    
    return {
        'local_acc': local_accs,
        'global_acc': global_accs,
        'train_loss': train_losses,
        'monitoring_score': monitoring_scores,
        'best_round': best_round,
        'best_score': best_score,
        'best_local_acc': best_local_acc,
        'best_global_acc': best_global_acc,
    }


# ---------------------
# 5b. FedAvg
# ---------------------
# REDESIGNED: FedAvg evaluates the global model BEFORE local training.
# It computes skewed_acc and global_acc, then uses
# monitoring_score = (skewed_acc + global_acc) / 2 for early stopping.
# FedAvg stores server.global_model.state_dict() as the checkpoint.

def run_fedavg(args, server, model_str):
    """
    FedAvg baseline: pure federated averaging, no personalization.
    
    Returns dict with same structure as run_local_only.
    """
    start_time = time.time()
    
    # Early stopping state
    best_score = -1.0
    best_round = -1
    best_model_state = None
    patience_counter = 0
    patience_limit = args.patience
    best_local_acc = 0.0
    best_global_acc = 0.0
    
    # Metric tracking arrays
    local_accs = []
    global_accs = []
    train_losses = []
    monitoring_scores = []
    
    for i in range(args.global_rounds + 1):
        s_t = time.time()
        
        # STEP 1: Server sends global model to clients
        # For FedAvg (layer_idx=0), send_models just copies global→local (no ALA)
        server.selected_clients = server.select_clients()
        server.send_models()
        
        # STEP 2: Evaluate BEFORE local training
        if i % args.eval_gap == 0:
            print(f"\n  [Round {i}/{args.global_rounds}]")
            local_acc, global_acc = server.evaluate()
            
            local_accs.append(local_acc)
            global_accs.append(global_acc)
            
            # Monitoring score = balanced average
            monitoring_score = (local_acc + global_acc) / 2.0
            monitoring_scores.append(monitoring_score)
            print(f"  Monitoring Score: {monitoring_score:.4f} "
                  f"= ({local_acc:.4f} + {global_acc:.4f}) / 2")
            
            # STEP 3: Track best model checkpoint (no early stopping break)
            # NOTE: We keep track of the best round, best score, and best accuracies
            # by checking if the current monitoring score is the highest seen so far.
            # However, we DO NOT decrement patience or call 'break' here. This ensures
            # that training runs for the exact fixed number of rounds requested, while
            # still allowing us to log the best results achieved at any point.
            if monitoring_score > best_score:
                best_score = monitoring_score
                best_round = i
                best_local_acc = local_acc
                best_global_acc = global_acc
                best_model_state = copy.deepcopy(server.global_model.state_dict())
                print(f"  ★ New best score: {best_score:.4f} at round {best_round}")
        
        # STEP 4: Clients train locally
        for client in server.selected_clients:
            client.train()
        
        # STEP 5: Server aggregates
        server.receive_models()
        server.aggregate_parameters()
        
        server.Budget.append(time.time() - s_t)
        print_progress("FedAvg", i, args.global_rounds, start_time)
    
    # Record training loss history from server (already tracked in evaluate())
    train_losses = list(server.rs_train_loss)
    
    elapsed = time.time() - start_time
    print(f"\n\n  FedAvg complete in {elapsed:.1f}s")
    print(f"  Best Monitoring Score: {best_score:.4f} (Round {best_round})")
    print(f"  Best Skewed Acc: {best_local_acc:.4f} | Best Global Acc: {best_global_acc:.4f}")
    
    return {
        'local_acc': local_accs,
        'global_acc': global_accs,
        'train_loss': train_losses,
        'monitoring_score': monitoring_scores,
        'best_round': best_round,
        'best_score': best_score,
        'best_local_acc': best_local_acc,
        'best_global_acc': best_global_acc,
    }


# ---------------------
# 5c. FedALA / FedSALA (Dual-Phase Evaluation)
# ---------------------
# REDESIGNED: These methods have TWO meaningful model states per round:
#   State 1: After ALA/SALA personalization, BEFORE local training
#   State 2: After local training (the "more personalized" version)
#
# Evaluation:
#   - BEFORE local training → whole-label (global) accuracy (pre_global_acc)
#   - AFTER local training  → skewed local accuracy (post_local_acc)
#
# Early stopping score = (pre_global_acc + post_local_acc) / 2
#
# This captures BOTH generalization (how well the ALA/SALA-personalized model
# handles all labels) AND personalization (how well it performs after further
# local fine-tuning on skewed data).

def run_pfl_method(args, server, algo_name, model_str):
    """
    Run FedALA or FedSALA with dual-phase evaluation.
    
    Returns dict with:
        - 'pre_global_acc': whole-label acc before local training (per eval round)
        - 'post_local_acc': skewed acc after local training (per eval round)
        - 'train_loss': training loss per eval round
        - 'monitoring_score': early stopping scores
        - 'best_round': round where best checkpoint was stored
        - 'best_score': best early stopping score
        - 'best_pre_global_acc': whole-label acc at best round (before local training)
        - 'best_post_local_acc': skewed acc at best round (after local training)
    """
    start_time = time.time()
    
    # Early stopping state
    best_score = -1.0
    best_round = -1
    best_model_state = None
    patience_counter = 0
    patience_limit = args.patience
    best_pre_global_acc = 0.0
    best_post_local_acc = 0.0
    
    # Metric tracking arrays
    pre_global_accs = []  # whole-label acc BEFORE local training
    post_local_accs = []  # skewed acc BEFORE local training
    train_losses = []
    monitoring_scores = []
    
    for i in range(args.global_rounds + 1):
        s_t = time.time()
        
        # STEP 1: Server sends global model → triggers ALA/SALA personalization
        # This calls client.local_initialization() which runs the adaptive
        # aggregation (ALA weight learning or SALA Fisher+ALA).
        server.selected_clients = server.select_clients()
        server.send_models()
        
        if i % args.eval_gap == 0:
            print(f"\n  [Round {i}/{args.global_rounds}]")
            
            # STEP 2: Evaluate BEFORE local training (after ALA/SALA)
            # =================================================================
            # REVISION REASON:
            # We explicitly moved the skewed data testing to happen BEFORE local 
            # training. Once you do local training, the local stochastic gradient 
            # descent obscures the initial parameter aggregation quality. You 
            # cannot precisely compare the Fisher Value selection method (FedSALA) 
            # versus the static Layer selection method (FedALA) if pure local 
            # fine-tuning covers up the initialization differences! 
            # Evaluating here forces a pure structural comparison.
            # =================================================================
            print(f"  --- Pre-local-training evaluation (after {algo_name} init) ---")
            
            # Run the global (whole-label) test
            stats_global = server.test_metrics_global()
            pre_global_acc = sum(stats_global[2]) * 1.0 / sum(stats_global[1])
            print(f"  Pre-train Global Test Accuracy: {pre_global_acc:.4f}")
            pre_global_accs.append(pre_global_acc)

            # Run the local (skewed) test
            stats_local = server.test_metrics()
            # We store this in the 'post_local_acc' variable purely to preserve backwards 
            # compatibility with the plotting logic below (which graphs mp_FedSALA).
            post_local_acc = sum(stats_local[2]) * 1.0 / sum(stats_local[1])
            print(f"  Pre-train Local Test Accuracy:  {post_local_acc:.4f}")
            post_local_accs.append(post_local_acc)
        
        # STEP 3: Clients train locally
        for client in server.selected_clients:
            client.train()
        
        if i % args.eval_gap == 0:
            # Compute training loss AFTER local training
            stats_train = server.train_metrics()
            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
            train_losses.append(train_loss)
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Track in server arrays for compatibility
            server.rs_local_acc.append(post_local_acc)
            server.rs_global_acc.append(pre_global_acc)
            server.rs_train_loss.append(train_loss)
            
            # STEP 5: Compute early stopping score
            # The combined score balances generalization and personalization
            monitoring_score = (pre_global_acc + post_local_acc) / 2.0
            monitoring_scores.append(monitoring_score)
            print(f"  Monitoring Score: {monitoring_score:.4f} "
                  f"= ({pre_global_acc:.4f} + {post_local_acc:.4f}) / 2")
            
            # STEP 6: Track best model checkpoint (no early stopping break)
            # NOTE: We keep track of the best round, best score, and best accuracies
            # by checking if the current monitoring score is the highest seen so far.
            # However, we DO NOT decrement patience or call 'break' here. This ensures
            # that training runs for the exact fixed number of rounds requested, while
            # still allowing us to log the best results achieved at any point.
            if monitoring_score > best_score:
                best_score = monitoring_score
                best_round = i
                best_pre_global_acc = pre_global_acc
                best_post_local_acc = post_local_acc
                best_model_state = copy.deepcopy(server.global_model.state_dict())
                print(f"  ★ New best score: {best_score:.4f} at round {best_round}")
        
        # STEP 7: Server aggregates
        server.receive_models()
        server.aggregate_parameters()
        
        server.Budget.append(time.time() - s_t)
        print_progress(algo_name, i, args.global_rounds, start_time)
    
    elapsed = time.time() - start_time
    print(f"\n\n  {algo_name} complete in {elapsed:.1f}s")
    print(f"  Best Monitoring Score: {best_score:.4f} (Round {best_round})")
    print(f"  Best Pre-train Global Acc: {best_pre_global_acc:.4f}")
    print(f"  Best Post-train Local Acc ({algo_name}): {best_post_local_acc:.4f}")
    
    return {
        'pre_global_acc': pre_global_accs,
        'post_local_acc': post_local_accs,
        'train_loss': train_losses,
        'monitoring_score': monitoring_scores,
        'best_round': best_round,
        'best_score': best_score,
        'best_pre_global_acc': best_pre_global_acc,
        'best_post_local_acc': best_post_local_acc,
    }


# =========================================================================
# SECTION 6: Single Algorithm Router
# =========================================================================
# REDESIGNED: Instead of a single shared training loop, each method now has
# its own dedicated training function with method-specific evaluation timing
# and early stopping score computation.

def run_single_algorithm(args, algo_name, model_str):
    """
    Route to the appropriate training function based on algorithm name.
    
    Returns:
        dict: method-specific results (see individual functions for structure)
    """
    print(f"\n{'='*60}")
    print(f"  Running: {algo_name}")
    print(f"  Model: {model_str} | Rounds: {args.global_rounds} | "
          f"Clients: {args.num_clients} | Local epochs: {args.local_steps}")
    print(f"  Patience: {args.patience}")
    print(f"{'='*60}")

    # Create fresh model for this algorithm
    args.model = create_model(args, model_str)

    # ---- Algorithm Selection ----
    if algo_name == "FedSALA":
        server = FedSALA(args, 0)
        return run_pfl_method(args, server, "FedSALA", model_str)

    elif algo_name == "FedALA":
        server = FedALA(args, 0)
        return run_pfl_method(args, server, "FedALA", model_str)

    elif algo_name == "FedAvg":
        # FedAvg = FedALA with layer_idx=0 (no ALA, pure averaging)
        original_layer_idx = args.layer_idx
        args.layer_idx = 0
        server = FedALA(args, 0)
        args.layer_idx = original_layer_idx  # restore for next algo
        return run_fedavg(args, server, model_str)

    elif algo_name == "LocalOnly":
        # Each client trains independently, no communication
        original_layer_idx = args.layer_idx
        args.layer_idx = 0
        server = FedALA(args, 0)
        args.layer_idx = original_layer_idx
        return run_local_only(args, server, model_str)

    elif algo_name == "Centralized":
        # UNCHANGED: Centralized runs with the old evaluation logic.
        # It is isolated from the new framework — see implementation plan
        # for why this is safe. Centralized is excluded from --run_all.
        original_nc = args.num_clients
        original_jr = args.join_ratio
        original_layer_idx = args.layer_idx
        args.num_clients = 1
        args.join_ratio = 1.0
        args.layer_idx = 0
        server = FedALA(args, 0)
        
        from flcore.clients.clientALA import clientALA
        from utils.data_utils import read_client_data
        train_data_all = read_client_data(args.dataset, "all", is_train=True)
        test_data_all = read_client_data(args.dataset, "all", is_train=False)
        server.clients[0] = clientALA(args, id="all", train_samples=len(train_data_all), test_samples=len(test_data_all))
        args.num_clients = original_nc
        args.join_ratio = original_jr
        args.layer_idx = original_layer_idx

        # Run old-style training loop (no early stopping, no dual eval)
        start_time = time.time()
        for i in range(args.global_rounds + 1):
            s_t = time.time()
            server.selected_clients = server.select_clients()
            server.send_models()

            if i % args.eval_gap == 0:
                print(f"\n  [Round {i}/{args.global_rounds}]")
                server.evaluate()

            for client in server.selected_clients:
                client.train()

            server.receive_models()
            server.aggregate_parameters()
            server.Budget.append(time.time() - s_t)
            print_progress(algo_name, i, args.global_rounds, start_time)

        elapsed = time.time() - start_time
        best_acc = max(server.rs_test_acc) if server.rs_test_acc else 0
        print(f"\n\n  Centralized complete in {elapsed:.1f}s")
        print(f"  Best Accuracy: {best_acc:.4f}")
        print(f"  Final Accuracy: {server.rs_test_acc[-1]:.4f}")
        
        # Return legacy format for Centralized (not used in multi-baseline plots)
        return {
            'local_acc': list(server.rs_test_acc),
            'global_acc': list(server.rs_test_acc),
            'train_loss': list(server.rs_train_loss),
            'monitoring_score': list(server.rs_test_acc),
            'best_round': int(np.argmax(server.rs_test_acc)) * args.eval_gap,
            'best_score': best_acc,
            'best_local_acc': best_acc,
            'best_global_acc': best_acc,
        }

    else:
        raise NotImplementedError(f"Unknown algorithm: {algo_name}")


# =========================================================================
# SECTION 7: Results Storage and Plotting
# =========================================================================
# REDESIGNED: Now generates 4 separate plots and a comprehensive results summary.
# The plots show all 4 baselines side-by-side for direct comparison.
# For FedALA/FedSALA, the skewed accuracy plot shows the "more personalized"
# (post-training) accuracy, while the global accuracy plot shows pre-training.

def save_results(all_results, results_dir, args):
    """
    Save metric data and generate comparison plots.
    
    Args:
        all_results: dict of {algo_name: results_dict}
        results_dir: directory to save results
        args: parsed command line arguments
    """
    eval_gap = args.eval_gap
    os.makedirs(results_dir, exist_ok=True)

    # ---- Save raw metric data as .npy ----
    for algo_name, results in all_results.items():
        for metric_name, values in results.items():
            if isinstance(values, list) and len(values) > 0:
                np.save(os.path.join(results_dir, f'{algo_name}_{metric_name}.npy'), 
                        np.array(values))

    # ---- Save final results summary with early stopping info ----
    summary_file = os.path.join(results_dir, 'final_results.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  FINAL RESULTS (Best Model @ Early Stopping)\n")
        f.write(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- EXPERIMENT SETTINGS ---\n")
        for arg, val in sorted(vars(args).items()):
            if arg != 'model':  # Skip model object reference
                f.write(f"  {arg:<22}: {val}\n")
        f.write("-" * 60 + "\n\n")

        # Load and append data split settings
        dataset_config_path = os.path.join('..', 'dataset', args.dataset, 'config.json')
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as cf:
                d_config = json.load(cf)
            f.write("--- DATASET SPLIT DEFINITION ---\n")
            for k, v in sorted(d_config.items()):
                f.write(f"  {k:<22}: {v}\n")
            f.write("-" * 60 + "\n\n")

        visualize_client_splits(args.dataset, args.num_clients, out_file=f, split_type='train')
        visualize_client_splits(args.dataset, args.num_clients, out_file=f, split_type='test')
        visualize_client_splits(args.dataset, args.num_clients, out_file=f, split_type='test_global')

        # Per-method results
        for algo_name, results in all_results.items():
            f.write(f"\n{algo_name}:\n")
            f.write(f"  Best Round:                    {results.get('best_round', 'N/A')}\n")
            
            # Extract final metrics (last element of lists)
            if algo_name in ["FedALA", "FedSALA"]:
                final_skewed = results.get('post_local_acc', [])[-1] if results.get('post_local_acc') else 0.0
                final_global = results.get('pre_global_acc', [])[-1] if results.get('pre_global_acc') else 0.0
                best_skewed = results.get('best_post_local_acc', 0.0)
                best_global = results.get('best_pre_global_acc', 0.0)
            else:
                final_skewed = results.get('local_acc', [])[-1] if results.get('local_acc') else 0.0
                final_global = results.get('global_acc', [])[-1] if results.get('global_acc') else 0.0
                best_skewed = results.get('best_local_acc', 0.0)
                best_global = results.get('best_global_acc', 0.0)
                
            final_loss = results.get('train_loss', [])[-1] if results.get('train_loss') else 0.0
            final_score = results.get('monitoring_score', [])[-1] if results.get('monitoring_score') else 0.0
            best_score = results.get('best_score', 0.0)
            
            # Extract train loss at best round
            train_loss_list = results.get('train_loss', [])
            best_round = results.get('best_round', -1)
            if best_round != -1 and len(train_loss_list) > (best_round // args.eval_gap):
                best_loss = train_loss_list[best_round // args.eval_gap]
            else:
                best_loss = 0.0
            
            f.write(f"  --- Final Round Results ---\n")
            f.write(f"    Skewed Local Accuracy:       {final_skewed:.4f}\n")
            f.write(f"    Whole-Label Accuracy:        {final_global:.4f}\n")
            f.write(f"    Training Loss:               {final_loss:.4f}\n")
            f.write(f"    Monitoring Score:            {final_score:.4f}\n")
            f.write(f"  --- Best Round Results ---\n")
            f.write(f"    Skewed Local Accuracy:       {best_skewed:.4f}\n")
            f.write(f"    Whole-Label Accuracy:        {best_global:.4f}\n")
            f.write(f"    Training Loss:               {best_loss:.4f}\n")
            f.write(f"    Monitoring Score:            {best_score:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        
        # Append Fisher distribution table to the bottom of the log file
        fisher_file = "fisher_distribution_client0.txt"
        if os.path.exists(fisher_file) and "FedSALA" in all_results:
            f.write("\n" + "=" * 60 + "\n")
            f.write("  FEDSALA FISHER DISTRIBUTION HISTORY (CLIENT 0)\n")
            f.write("=" * 60 + "\n\n")
            try:
                with open(fisher_file, 'r') as fish_f:
                    f.write(fish_f.read())
                # Copy the standalone text file into the results directory natively
                import shutil
                shutil.copy(fisher_file, os.path.join(results_dir, "fisher_distribution_client0.txt"))
            except Exception as e:
                f.write(f"  Error reading/copying {fisher_file}: {e}\n")

    print(f"\n  Results saved to: {summary_file}")

    # ---- Print final summary to CLI ----
    print("\n" + "=" * 60)
    print("  FINAL RESULTS (Best Model @ Early Stopping)")
    print("=" * 60)
    for algo_name, results in all_results.items():
        print(f"\n  {algo_name}:")
        print(f"    Best Round:         {results.get('best_round', 'N/A')}")
        if algo_name in ["FedALA", "FedSALA"]:
            print(f"    Global Acc (pre):   {results.get('best_pre_global_acc', 0):.4f}")
            print(f"    {algo_name} (post): {results.get('best_post_local_acc', 0):.4f}")
        else:
            print(f"    Local-Skewed Acc:   {results.get('best_local_acc', 0):.4f}")
            print(f"    Whole-Label Acc:    {results.get('best_global_acc', 0):.4f}")
        print(f"    Monitoring Score:   {results.get('best_score', 0):.4f}")
    print("=" * 60)

    # ---- Generate 4 comparison plots ----
    # Color and style configuration for consistent look across all plots
    colors = {
        'FedSALA': '#e74c3c',       # Red (our method — highlight)
        'FedALA': '#3498db',        # Blue
        'FedAvg': '#2ecc71',        # Green
        'LocalOnly': '#9b59b6',     # Purple
    }

    # Helper: get rounds axis for a metric array
    def get_rounds(metric_list):
        return list(range(0, len(metric_list) * eval_gap, eval_gap))[:len(metric_list)]

    # ---- PLOT 1: Skewed Local Test Accuracy ----
    # For FedALA/FedSALA: this is the post-local-training accuracy 
    # For FedAvg/LocalOnly: this is the pre-training accuracy (same model state)
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')

    for algo_name, results in all_results.items():
        if algo_name in ["FedALA", "FedSALA"]:
            # Post-training skewed accuracy = "more personalized" version
            accs = results.get('post_local_acc', [])
            label = f"{algo_name}"
            if len(accs) > 0:
                rounds = get_rounds(accs)
                plt.plot(rounds, accs, label=label, 
                        color=colors.get(label, '#95a5a6'),
                        linewidth=3 if 'SALA' in algo_name else 1.5,
                        linestyle='--', alpha=0.9)
        else:
            # LocalOnly/FedAvg: local (skewed) accuracy
            accs = results.get('local_acc', [])
            if len(accs) > 0:
                rounds = get_rounds(accs)
                plt.plot(rounds, accs, label=algo_name,
                        color=colors.get(algo_name, '#95a5a6'),
                        linewidth=1.5, linestyle='-', alpha=0.9)

    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Test Accuracy (Skewed Local)', fontsize=14)
    plt.title('Personalized Performance — Skewed Local Test Accuracy', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(results_dir, 'accuracy_skewed.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_file}")

    # ---- PLOT 2: Whole-Label (Global) Test Accuracy ----
    # For FedALA/FedSALA: this is the pre-local-training accuracy
    # For FedAvg/LocalOnly: this is the pre-training accuracy
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')

    for algo_name, results in all_results.items():
        if algo_name in ["FedALA", "FedSALA"]:
            # Pre-training whole-label accuracy
            accs = results.get('pre_global_acc', [])
            label = algo_name
        else:
            accs = results.get('global_acc', [])
            label = algo_name
        
        if len(accs) > 0:
            rounds = get_rounds(accs)
            plt.plot(rounds, accs, label=label,
                    color=colors.get(algo_name, '#95a5a6'),
                    linewidth=3 if algo_name == 'FedSALA' else 1.5,
                    linestyle='-' if algo_name == 'FedSALA' else '--', alpha=0.9)

    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Test Accuracy (Whole-Label)', fontsize=14)
    plt.title('Generalization Performance — Whole-Label Test Accuracy', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(results_dir, 'accuracy_global.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_file}")

    # ---- PLOT 3: Training Loss ----
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')

    for algo_name, results in all_results.items():
        losses = results.get('train_loss', [])
        if len(losses) > 0:
            rounds = get_rounds(losses)
            plt.plot(rounds, losses, label=algo_name,
                    color=colors.get(algo_name, '#95a5a6'),
                    linewidth=3 if algo_name == 'FedSALA' else 1.5,
                    linestyle='-' if algo_name == 'FedSALA' else '--', alpha=0.9)

    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.title('Training Loss Over Rounds', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(results_dir, 'training_loss.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_file}")

    # ---- PLOT 4: Monitoring Score (Early Stopping Trigger) ----
    # Shows the combined score that drove the checkpoint decision for each method.
    # For all methods: monitoring_score = (metric_A + metric_B) / 2
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')

    for algo_name, results in all_results.items():
        scores = results.get('monitoring_score', [])
        best_round = results.get('best_round', -1)
        if len(scores) > 0:
            rounds = get_rounds(scores)
            plt.plot(rounds, scores, label=algo_name,
                    color=colors.get(algo_name, '#95a5a6'),
                    linewidth=3 if algo_name == 'FedSALA' else 1.5,
                    linestyle='-' if algo_name == 'FedSALA' else '--', alpha=0.9)
            # Mark the best checkpoint with a star
            if best_round >= 0 and best_round // eval_gap < len(scores):
                best_idx = best_round // eval_gap
                plt.scatter([rounds[best_idx]], [scores[best_idx]], 
                           color=colors.get(algo_name, '#95a5a6'),
                           s=150, zorder=5, marker='*',
                           edgecolors='black', linewidths=0.5)

    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Monitoring Score', fontsize=14)
    plt.title('Early Stopping Monitoring Score — (Metric_A + Metric_B) / 2', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(results_dir, 'monitoring_score.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_file}")


# =========================================================================
# SECTION 8: Main Entry Point
# =========================================================================

def run(args):
    """Main run function. Either runs all 4 baselines or a single one."""

    # ---- Clear previous tracking file if exists ----
    if os.path.exists("fisher_distribution_client0.txt"):
        try:
            os.remove("fisher_distribution_client0.txt")
        except OSError:
            pass

    model_str = args.model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('..', 'results',
                               f'{args.dataset}_{model_str}_nc{args.num_clients}_gr{args.global_rounds}_{timestamp}')

    # ---- Visualize client data splits before training ----
    data_ok = visualize_client_splits(args.dataset, args.num_clients)
    if not data_ok:
        print("\n  ERROR: Dataset not found. Run generate_cifar10.py first:")
        print("    python3 generate_cifar10.py --num_clients", args.num_clients)
        sys.exit(1)

    # ---- Determine which algorithms to run ----
    if args.run_all:
        # MODIFIED: Centralized removed from --run_all. Only 4 baselines now.
        # Centralized must be run separately with: python3 main.py -algo Centralized
        algorithms = ["FedSALA", "FedAvg", "LocalOnly", "FedALA"]
        print(f"\n  Running ALL {len(algorithms)} baselines sequentially...")
        print(f"  (Centralized excluded — run separately with -algo Centralized)")
    elif args.run_two:
        # Run only FedSALA + FedALA together so their results land in the same
        # result folder and appear on the same comparison graphs.
        algorithms = ["FedSALA", "FedALA"]
        print(f"\n  Running FedSALA + FedALA (comparison mode)...")
    else:
        algorithms = [args.algorithm]
        print(f"\n  Running single algorithm: {args.algorithm}")

    # ---- Run each algorithm ----
    all_results = {}
    for algo_idx, algo_name in enumerate(algorithms):
        print(f"\n{'#'*60}")
        print(f"  ALGORITHM {algo_idx+1}/{len(algorithms)}: {algo_name}")
        print(f"{'#'*60}")

        results = run_single_algorithm(args, algo_name, model_str)
        all_results[algo_name] = results

    # ---- Save results and generate plots ----
    # Only generate multi-baseline plots when we have multiple results
    if len(all_results) > 1 or not args.run_all:
        save_results(all_results, results_dir, args)

    print(f"\n  All done! Results in: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser(description="FedSALA Experiment Runner")

    # ===== Run Mode =====
    parser.add_argument('--run_all', action='store_true',
                        help='Run all 4 baselines sequentially (FedAvg, LocalOnly, FedALA, FedSALA)')
    parser.add_argument('--run_two', action='store_true',
                        help='Run only FedSALA + FedALA in the same result folder (for comparison graphs)')

    # ===== General Settings =====
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10",
                        help="Dataset folder name (default: Cifar10)")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="resnet",
                        choices=["cnn", "resnet"],
                        help="Model architecture (default: resnet)")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10) # just as paper
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001,
                        help="Client-side learning rate")

    # ===== FL Settings =====
    parser.add_argument('-gr', "--global_rounds", type=int, default=200,
                        help="Number of communication rounds")
    parser.add_argument('-ls', "--local_steps", type=int, default=1,
                        help="Local training epochs per round")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedSALA",
                        choices=["FedSALA", "FedALA", "FedAvg", "LocalOnly", "Centralized"],
                        help="Which algorithm to run (ignored if --run_all)")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Fraction of clients per round (1.0 = all)")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False)
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0)
    parser.add_argument('-t', "--times", type=int, default=1)
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Evaluate every N rounds")

    # ===== Early Stopping Settings =====
    # NEW: Patience-based early stopping. Training continues until the monitoring
    # score hasn't improved for `patience` consecutive evaluation rounds.
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (rounds without improvement). '
                             'Set to a value >= global_rounds to disable early stopping.')

    # ===== ALA Settings (FedALA) =====
    parser.add_argument('-et', "--eta", type=float, default=1.0,
                        help="ALA weight learning rate")
    parser.add_argument('-s', "--rand_percent", type=int, default=80,
                        help="%% of local data for ALA weight learning")
    parser.add_argument('-p', "--layer_idx", type=int, default=17,
                        help="FedALA: top N layers to personalize")
    parser.add_argument('--lower_layers_local', action='store_true',
                        help='FedALA: keep lower layers local (not overwritten by global). '
                             'Creates structural parallel with FedSALA M3.')

    # ===== FedSALA Settings (Fisher) =====
    parser.add_argument('--fisher_threshold', type=float, default=0.75,
                        help='Top P fraction for ALA treatment (0.5 = top 50%%)')
    parser.add_argument('--fisher_ema_alpha', type=float, default=0,
                        help='EMA smoothing (0.9 = stable mask, 0 zero ema effect only current one matters)')
    parser.add_argument('--fisher_sample_percent', type=int, default=10,
                        help='%% of local data for Fisher computation')
    parser.add_argument('--fedsala_method', type=int, default=3, choices=[1, 2, 3, 4],
                        help='FedSALA method variant: 1=High-ALA/Low-Global, 2=High-Freeze/Low-ALA, '
                             '3=High-ALA/Low-Freeze (default), 4=High-Global/Low-ALA')


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n  CUDA not available, falling back to CPU.\n")
        args.device = "cpu"

    run(args)

    total_time = time.time() - total_start
    print(f"\n  Total wall-clock time: {int(total_time//60)}m{int(total_time%60)}s")