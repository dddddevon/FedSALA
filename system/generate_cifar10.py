"""
generate_cifar10.py — Flexible Pathological Non-IID Data Generator
====================================================================

WHAT THIS DOES:
    Downloads CIFAR-10 and splits it across N clients in a pathological non-IID way.
    Supports:
      - Uniform label assignment (all clients get same # of labels, equal samples)
      - Ratio-based allocation (e.g. 4:2:1 within each client)
      - Mixed client types via JSON config (different groups with different label counts)
    
    Generates THREE data splits:
      1. Training data  → ../dataset/Cifar10/train/{idx}.npz
      2. Skewed test    → ../dataset/Cifar10/test/{idx}.npz   (same distribution, 1/5 size)
      3. Global test    → ../dataset/Cifar10/test_global/{idx}.npz (all 10 labels, balanced)

HOW TO USE:

    === CLI Flags (Uniform — all clients same config) ===

    # Equal samples per label (backward compatible with old script)
    python3 generate_cifar10.py --num_clients 10 --labels_per_client 2 --samples_per_label 1000

    # Ratio-based allocation
    python3 generate_cifar10.py --num_clients 10 --labels_per_client 3 --ratio 4:2:1 --total_per_client 280

    # Camera-trap proxy
    python3 generate_cifar10.py --num_clients 15 --labels_per_client 2 --ratio 6:1 --total_per_client 210

    === JSON Config (Mixed — different client groups) ===

    python3 generate_cifar10.py --config scenarios/hospital_mixed.json

    JSON format:
    {
      "client_groups": [
        {"count": 4, "labels_per_client": 4, "ratio": [4,3,2,1], "total_per_client": 400},
        {"count": 6, "labels_per_client": 2, "ratio": [3,1],     "total_per_client": 200}
      ],
      "seed": 42
    }

    If --config is provided, it overrides all other CLI flags.

OUTPUT:
    Creates ../dataset/Cifar10/{train,test,test_global}/ directories
    with one .npz file per client.
"""

import os
import json
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict


# =========================================================================
# SECTION 1: CIFAR-10 Data Loading
# =========================================================================

def load_cifar10():
    """
    Download CIFAR-10 and organize into per-class pools.
    Returns:
        train_by_class: dict mapping class_id → list of image arrays
        test_by_class:  dict mapping class_id → list of image arrays
    """
    print("Downloading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='../dataset/raw', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='../dataset/raw', train=False, download=True, transform=transform)
    
    print("Organizing data by class...")
    train_by_class = defaultdict(list)
    for img, label in trainset:
        train_by_class[label].append(img.numpy())
    
    test_by_class = defaultdict(list)
    for img, label in testset:
        test_by_class[label].append(img.numpy())
    
    return train_by_class, test_by_class


# =========================================================================
# SECTION 2: Label Assignment — Round-Robin with Shuffle
# =========================================================================
# NEW: This replaces the old shard-based assignment. It creates a "deck" of
# class IDs, shuffles it, and deals cards to each client. This ensures each
# class appears in roughly the same number of clients with minimal overlap.
# See implementation plan for detailed walkthrough.
# =========================================================================

def assign_labels_round_robin(client_specs, num_classes=10, seed=42):
    """
    Assign labels to clients using round-robin with shuffle.
    
    Args:
        client_specs: list of dicts, one per client. Each has 'labels_per_client'.
        num_classes: total number of classes (10 for CIFAR-10)
        seed: random seed for reproducibility
    
    Returns:
        list of lists: client_labels[i] = [label_0, label_1, ...] for client i
    
    Algorithm:
        1. Count total slots = sum of labels_per_client across all clients
        2. Each class gets floor(total_slots / num_classes) slots, with remainder
           distributed to the first few classes (so some classes appear in 1 more client)
        3. Build a "deck" of class IDs, shuffle it
        4. Deal cards to each client according to their labels_per_client count
    """
    np.random.seed(seed)
    
    num_clients = len(client_specs)
    total_slots = sum(spec['labels_per_client'] for spec in client_specs)
    
    # Step 1: Calculate how many clients each class should appear in
    # With 30 slots / 10 classes = 3 slots each (each class in 3 clients)
    base_slots = total_slots // num_classes
    remainder = total_slots % num_classes
    
    # remaining_slots[class_id] = how many more clients still need this class
    remaining_slots = {}
    for class_id in range(num_classes):
        remaining_slots[class_id] = base_slots + (1 if class_id < remainder else 0)
    
    # Step 2: Greedy assignment — for each client, pick labels from classes
    # with the most remaining slots (prevents duplicate labels per client)
    # Shuffle client order so no client is systematically advantaged
    client_order = list(range(num_clients))
    np.random.shuffle(client_order)
    
    client_labels = [None] * num_clients
    
    for client_id in client_order:
        n = client_specs[client_id]['labels_per_client']
        
        # Sort available classes by remaining slots (descending), break ties randomly
        available = list(remaining_slots.keys())
        np.random.shuffle(available)  # randomize within same-slot-count groups
        available.sort(key=lambda c: remaining_slots[c], reverse=True)
        
        # Take the top n classes that still have slots > 0
        chosen = []
        for class_id in available:
            if remaining_slots[class_id] > 0 and len(chosen) < n:
                chosen.append(class_id)
        
        # Edge case: if not enough classes have remaining slots (shouldn't happen
        # with correct total_slots calculation, but safety net)
        if len(chosen) < n:
            # Fill from any class, even if over-assigned
            all_classes = list(range(num_classes))
            np.random.shuffle(all_classes)
            for c in all_classes:
                if c not in chosen and len(chosen) < n:
                    chosen.append(c)
        
        client_labels[client_id] = chosen
        
        # Decrement remaining slots for chosen classes
        for c in chosen:
            remaining_slots[c] = remaining_slots.get(c, 0) - 1
    
    # Print label coverage summary
    print(f"\n  Label Assignment Summary:")
    for class_id in range(num_classes):
        clients_with_class = [i for i, labels in enumerate(client_labels) if class_id in labels]
        print(f"    Class {class_id}: assigned to {len(clients_with_class)} clients → {clients_with_class}")
    
    return client_labels


# =========================================================================
# SECTION 3: Sample Allocation — Pool + Cursor
# =========================================================================
# NEW: This replaces the old fixed-shard slicing. Each CIFAR class has a pool
# of shuffled images. A cursor per class tracks consumption. When multiple
# clients share a class, each picks up where the previous left off, ensuring
# zero overlap. See implementation plan for detailed walkthrough.
# =========================================================================

def allocate_samples(client_specs, client_labels, train_by_class, test_by_class, 
                     num_classes=10, seed=42):
    """
    Allocate actual data samples to each client based on their label assignment
    and ratio/count specification.
    
    Args:
        client_specs: list of dicts, each has 'labels_per_client', 'ratio', 
                      'total_per_client' OR 'samples_per_label'
        client_labels: from assign_labels_round_robin()
        train_by_class: dict of class_id → list of train images
        test_by_class: dict of class_id → list of test images
    
    Returns:
        client_train_data: dict of client_id → {'x': [...], 'y': [...]}
        client_test_data:  dict of client_id → {'x': [...], 'y': [...]}
    """
    np.random.seed(seed + 1)  # Different seed from label assignment for independence
    
    num_clients = len(client_specs)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Shuffle each class pool once so all clients draw from a randomized order
    train_pools = {}
    test_pools = {}
    for class_id in range(num_classes):
        train_arr = np.array(train_by_class[class_id])
        np.random.shuffle(train_arr)
        train_pools[class_id] = train_arr
        
        test_arr = np.array(test_by_class[class_id])
        np.random.shuffle(test_arr)
        test_pools[class_id] = test_arr
    
    # Cursors track how far into each class pool we've consumed
    # This guarantees zero overlap between clients sharing the same class
    train_cursors = {c: 0 for c in range(num_classes)}
    test_cursors = {c: 0 for c in range(num_classes)}
    
    # ---- Pre-compute per-client, per-label sample counts ----
    # This is where ratio → concrete counts conversion happens
    client_sample_counts = []  # list of dicts: {label_id: train_count}
    
    for client_id in range(num_clients):
        spec = client_specs[client_id]
        labels = client_labels[client_id]
        counts = {}
        
        if 'samples_per_label' in spec:
            # UNIFORM MODE: every label gets the same count
            for label in labels:
                counts[label] = spec['samples_per_label']
        else:
            # RATIO MODE: distribute total_per_client according to ratio
            ratio = spec['ratio']
            total = spec['total_per_client']
            ratio_sum = sum(ratio)
            
            # Sort ratio descending — first dealt label gets largest share
            sorted_ratio = sorted(ratio, reverse=True)
            
            # Assign ratio values to labels in the order they were dealt
            remaining = total
            for i, label in enumerate(labels):
                if i < len(sorted_ratio) - 1:
                    # Integer division for all but the last to avoid rounding errors
                    count = int(total * sorted_ratio[i] / ratio_sum)
                    counts[label] = count
                    remaining -= count
                else:
                    # Last label gets whatever remains (handles rounding)
                    counts[label] = remaining
        
        client_sample_counts.append(counts)
    
    # ---- Check pool capacity before drawing ----
    # Sum up total demand per class across all clients
    demand_per_class = defaultdict(int)
    for client_id in range(num_clients):
        for label, count in client_sample_counts[client_id].items():
            demand_per_class[label] += count
    
    for class_id in range(num_classes):
        train_available = len(train_pools[class_id])
        train_demand = demand_per_class[class_id]
        # Skewed test is 1/5 of training, drawn from test pool
        test_demand = sum(
            client_sample_counts[cid].get(class_id, 0) // 5
            for cid in range(num_clients)
        )
        test_available = len(test_pools[class_id])
        
        if train_demand > train_available:
            print(f"  ⚠ WARNING: Class {class_id} ({class_names[class_id]}): "
                  f"demand={train_demand} > available={train_available} training samples!")
        if test_demand > test_available:
            print(f"  ⚠ WARNING: Class {class_id} ({class_names[class_id]}): "
                  f"test demand={test_demand} > available={test_available} test samples!")
    
    # ---- Draw samples using cursors ----
    client_train_data = defaultdict(lambda: {'x': [], 'y': []})
    client_test_data = defaultdict(lambda: {'x': [], 'y': []})
    
    for client_id in range(num_clients):
        counts = client_sample_counts[client_id]
        
        for label, train_count in counts.items():
            # --- Training samples ---
            start = train_cursors[label]
            end = min(start + train_count, len(train_pools[label]))
            actual_train = end - start
            
            if actual_train > 0:
                client_train_data[client_id]['x'].append(train_pools[label][start:end])
                client_train_data[client_id]['y'].extend([label] * actual_train)
                train_cursors[label] = end
            
            # --- Skewed test samples (1/5 of training count) ---
            test_count = train_count // 5
            t_start = test_cursors[label]
            t_end = min(t_start + test_count, len(test_pools[label]))
            actual_test = t_end - t_start
            
            if actual_test > 0:
                client_test_data[client_id]['x'].append(test_pools[label][t_start:t_end])
                client_test_data[client_id]['y'].extend([label] * actual_test)
                test_cursors[label] = t_end
    
    return client_train_data, client_test_data, client_sample_counts


# =========================================================================
# SECTION 4: Save Data + Display
# =========================================================================

def save_client_data(client_train_data, client_test_data, num_clients, seed):
    """
    Save train and test .npz files and print distribution summary.
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    output_dir = '../dataset/Cifar10'
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING DATA")
    print(f"{'='*70}")
    print(f"  {'Client':<10} {'Train':<8} {'Test':<8} {'Classes':<50}")
    print(f"  {'-'*66}")
    
    for client_id in range(num_clients):
        # Concatenate all label chunks for this client
        x_train = np.concatenate(client_train_data[client_id]['x'], axis=0).astype(np.float32)
        y_train = np.array(client_train_data[client_id]['y'], dtype=np.int64)
        
        x_test = np.concatenate(client_test_data[client_id]['x'], axis=0).astype(np.float32)
        y_test = np.array(client_test_data[client_id]['y'], dtype=np.int64)
        
        # Shuffle within each client for training randomness
        np.random.seed(seed + client_id)
        train_perm = np.random.permutation(len(x_train))
        x_train, y_train = x_train[train_perm], y_train[train_perm]
        
        test_perm = np.random.permutation(len(x_test))
        x_test, y_test = x_test[test_perm], y_test[test_perm]
        
        # Save in the format data_utils.py expects: {'x': array, 'y': array}
        np.savez_compressed(
            os.path.join(train_dir, f'{client_id}.npz'),
            data={'x': x_train, 'y': y_train}
        )
        np.savez_compressed(
            os.path.join(test_dir, f'{client_id}.npz'),
            data={'x': x_test, 'y': y_test}
        )
        
        # Display class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_info = ', '.join([f'{class_names[c]}({cnt})' for c, cnt in zip(unique, counts)])
        print(f"  {client_id:<10} {len(x_train):<8} {len(x_test):<8} {class_info}")
    
    print(f"{'='*70}")
    print(f"\n  Saved to: {os.path.abspath(output_dir)}")
    print(f"    Train: {train_dir}/{{0..{num_clients-1}}}.npz")
    print(f"    Test:  {test_dir}/{{0..{num_clients-1}}}.npz")
    
    return output_dir


# =========================================================================
# SECTION 5: Whole-Label Global Test Shards
# =========================================================================
# MODIFIED: Changed from global-shuffle-then-slice to STRATIFIED per-class
# allocation. Now every client gets EXACTLY the same number of samples for
# every label — perfectly balanced, zero skew, non-overlapping.
#
# Old approach: shuffle all 10k test images, slice into N equal chunks.
#   → Roughly balanced but not exact (e.g., airplane=112, bird=86)
#
# New approach: for each class independently, shuffle its 1000 images and
#   deal exactly (1000 // N) to each client.
#   → Perfectly balanced (e.g., airplane=100, bird=100 for 10 clients)
# =========================================================================

def generate_whole_label_test_shards(test_by_class, num_clients, num_classes=10, seed=42):
    """
    Generate whole-label test shards where each client gets samples from ALL 10 classes.
    
    PERFECTLY BALANCED: Every client gets exactly (1000 // num_clients) samples
    per class. All clients have identical label distributions.
    NON-OVERLAPPING: Each image is assigned to exactly one client.
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    output_dir = '../dataset/Cifar10'
    test_global_dir = os.path.join(output_dir, 'test_global')
    os.makedirs(test_global_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating whole-label global test shards (stratified)...")
    
    # CIFAR-10 test set has 1,000 images per class
    samples_per_class = len(test_by_class[0])  # should be 1000
    samples_per_class_per_client = samples_per_class // num_clients
    
    print(f"  {samples_per_class} test images per class")
    print(f"  {samples_per_class_per_client} per class per client "
          f"({samples_per_class_per_client} × {num_classes} = "
          f"{samples_per_class_per_client * num_classes} total per client)")
    
    # Prepare per-client accumulators
    client_global_x = {cid: [] for cid in range(num_clients)}
    client_global_y = {cid: [] for cid in range(num_clients)}
    
    # Stratified allocation: for EACH class independently,
    # shuffle its pool and deal equal slices to each client
    np.random.seed(seed)
    
    for class_id in range(num_classes):
        # Get all test images for this class and shuffle them
        class_images = np.array(test_by_class[class_id]).astype(np.float32)
        perm = np.random.permutation(len(class_images))
        class_images = class_images[perm]
        
        # Deal exactly (samples_per_class_per_client) images to each client
        for client_id in range(num_clients):
            start = client_id * samples_per_class_per_client
            end = start + samples_per_class_per_client
            
            client_global_x[client_id].append(class_images[start:end])
            client_global_y[client_id].extend([class_id] * samples_per_class_per_client)
    
    # Save each client's global test shard
    for client_id in range(num_clients):
        shard_x = np.concatenate(client_global_x[client_id], axis=0)
        shard_y = np.array(client_global_y[client_id], dtype=np.int64)
        
        # Shuffle within the shard so labels aren't in order during evaluation
        perm = np.random.permutation(len(shard_x))
        shard_x, shard_y = shard_x[perm], shard_y[perm]
        
        np.savez_compressed(
            os.path.join(test_global_dir, f'{client_id}.npz'),
            data={'x': shard_x, 'y': shard_y}
        )
        
        unique, counts = np.unique(shard_y, return_counts=True)
        class_info = ', '.join([f'{class_names[c]}({cnt})' for c, cnt in zip(unique, counts)])
        print(f"  Client {client_id:<2} Global Test: {len(shard_x):<5} samples -> {class_info}")
    
    print(f"Saved to: {test_global_dir}/{{0..{num_clients-1}}}.npz")
    print("=" * 60)


# =========================================================================
# SECTION 6: Config Saving
# =========================================================================
# NEW: Save enriched config.json that includes per-client label assignments
# and sample counts, so final_results.txt can reference the exact split.
# =========================================================================

def save_config(client_specs, client_labels, client_sample_counts, num_clients, seed):
    """
    Save config.json with full per-client details for reproducibility.
    """
    output_dir = '../dataset/Cifar10'
    
    # Build per-client assignment record
    client_assignments = {}
    for client_id in range(num_clients):
        labels = [int(l) for l in client_labels[client_id]]
        train_samples = {str(k): v for k, v in client_sample_counts[client_id].items()}
        test_samples = {str(k): v // 5 for k, v in client_sample_counts[client_id].items()}
        client_assignments[str(client_id)] = {
            'labels': labels,
            'train_samples': train_samples,
            'test_samples': test_samples
        }
    
    # Determine partition description
    all_same = all(
        spec.get('labels_per_client') == client_specs[0].get('labels_per_client')
        for spec in client_specs
    )
    
    config = {
        'dataset': 'Cifar10',
        'num_clients': num_clients,
        'num_classes': 10,
        'non_iid': True,
        'partition': 'pathological',
        'mixed_types': not all_same,
        'seed': seed,
        'client_assignments': client_assignments
    }
    
    # Add uniform-mode summary if all clients are the same
    if all_same:
        spec = client_specs[0]
        config['labels_per_client'] = spec['labels_per_client']
        if 'samples_per_label' in spec:
            config['samples_per_label'] = spec['samples_per_label']
        if 'ratio' in spec:
            config['ratio'] = spec['ratio']
            config['total_per_client'] = spec['total_per_client']
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Config saved to: {os.path.join(output_dir, 'config.json')}")


# =========================================================================
# SECTION 7: Main Orchestrator
# =========================================================================
# NEW: This is the single entry point that handles both CLI and JSON config,
# builds client_specs, then calls the label assignment → sample allocation
# → save pipeline.
# =========================================================================

def generate_flexible_noniid(client_specs, seed=42):
    """
    Main orchestrator for flexible non-IID data generation.
    
    Args:
        client_specs: list of dicts, one per client. Each has:
            - 'labels_per_client': int
            - Either 'samples_per_label': int (uniform mode)
            - Or 'ratio': list[int] + 'total_per_client': int (ratio mode)
        seed: random seed
    """
    num_clients = len(client_specs)
    
    # ---- Load CIFAR-10 ----
    train_by_class, test_by_class = load_cifar10()
    
    # ---- Step 1: Assign labels using round-robin with shuffle ----
    print(f"\nPathological Non-IID Split:")
    print(f"  {num_clients} clients")
    
    client_labels = assign_labels_round_robin(client_specs, num_classes=10, seed=seed)
    
    # ---- Step 2: Allocate data samples using pool + cursor ----
    client_train_data, client_test_data, client_sample_counts = allocate_samples(
        client_specs, client_labels, train_by_class, test_by_class,
        num_classes=10, seed=seed
    )
    
    # ---- Step 3: Save training + skewed test data ----
    save_client_data(client_train_data, client_test_data, num_clients, seed)
    
    # ---- Step 4: Generate whole-label global test shards ----
    generate_whole_label_test_shards(test_by_class, num_clients, num_classes=10, seed=seed)
    
    # ---- Step 5: Save config.json ----
    save_config(client_specs, client_labels, client_sample_counts, num_clients, seed)
    
    return num_clients


# =========================================================================
# SECTION 8: CLI Argument Parsing + Config Loading
# =========================================================================
# NEW: Supports three modes:
#   1. --config file.json           → mixed client types from JSON
#   2. --ratio R --total_per_client T → uniform ratio mode from CLI
#   3. --samples_per_label N        → uniform equal mode from CLI (backward compat)
# =========================================================================

def build_client_specs_from_args(args):
    """
    Convert CLI arguments into a list of per-client spec dicts.
    
    Returns:
        list of dicts, one per client
    """
    specs = []
    for _ in range(args.num_clients):
        spec = {'labels_per_client': args.labels_per_client}
        
        if args.ratio:
            # Ratio mode: parse "4:2:1" into [4, 2, 1]
            ratio = [int(r) for r in args.ratio.split(':')]
            if len(ratio) != args.labels_per_client:
                raise ValueError(
                    f"Ratio has {len(ratio)} values but labels_per_client is "
                    f"{args.labels_per_client}. They must match."
                )
            spec['ratio'] = ratio
            spec['total_per_client'] = args.total_per_client
        else:
            # Uniform mode: equal samples per label
            spec['samples_per_label'] = args.samples_per_label
        
        specs.append(spec)
    
    return specs


def build_client_specs_from_config(config_path):
    """
    Load a JSON config file and flatten client_groups into per-client specs.
    
    JSON format:
    {
      "client_groups": [
        {"count": 4, "labels_per_client": 4, "ratio": [4,3,2,1], "total_per_client": 400},
        {"count": 6, "labels_per_client": 2, "ratio": [3,1], "total_per_client": 200}
      ],
      "seed": 42
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    specs = []
    for group in config['client_groups']:
        count = group['count']
        for _ in range(count):
            spec = {
                'labels_per_client': group['labels_per_client'],
            }
            if 'ratio' in group:
                spec['ratio'] = group['ratio']
                spec['total_per_client'] = group['total_per_client']
            elif 'samples_per_label' in group:
                spec['samples_per_label'] = group['samples_per_label']
            else:
                raise ValueError(f"Group must have 'ratio'+'total_per_client' or 'samples_per_label': {group}")
            specs.append(spec)
    
    seed = config.get('seed', 42)
    return specs, seed


# =========================================================================
# SECTION 9: Entry Point
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate CIFAR-10 pathological non-IID data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

=== CLI Flags (Uniform — all clients same config) ===

  # Equal samples per label (backward compatible):
  python3 generate_cifar10.py --num_clients 10 --labels_per_client 2 --samples_per_label 1000

  # Remote sensing proxy (3 labels, 4:2:1 ratio, 280 total/client):
  python3 generate_cifar10.py --num_clients 10 --labels_per_client 3 --ratio 4:2:1 --total_per_client 280

  # Hospital proxy (3 labels, 3:2:1 ratio, 300 total/client):
  python3 generate_cifar10.py --num_clients 10 --labels_per_client 3 --ratio 3:2:1 --total_per_client 300

  # Camera-trap proxy (2 labels, 6:1 ratio, 210 total/client):
  python3 generate_cifar10.py --num_clients 15 --labels_per_client 2 --ratio 6:1 --total_per_client 210

=== JSON Config (Mixed — different client groups) ===

  # Hospital with mixed types (4 general + 6 specialty):
  python3 generate_cifar10.py --config scenarios/hospital_mixed.json

  JSON format:
  {
    "client_groups": [
      {"count": 4, "labels_per_client": 4, "ratio": [4,3,2,1], "total_per_client": 400},
      {"count": 6, "labels_per_client": 2, "ratio": [3,1],     "total_per_client": 200}
    ],
    "seed": 42
  }

  If --config is provided, it overrides all other CLI flags.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    )
    
    # JSON config mode
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file for mixed client types. '
                             'Overrides all other flags if provided.')
    
    # CLI uniform mode
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients (default: 10)')
    parser.add_argument('--labels_per_client', type=int, default=2,
                        help='Labels per client (default: 2). '
                             'Alias: --classes_per_client')
    parser.add_argument('--classes_per_client', type=int, default=None,
                        help='Alias for --labels_per_client (backward compat)')
    
    # Uniform equal mode
    parser.add_argument('--samples_per_label', type=int, default=500,
                        help='Samples per label in uniform mode (default: 500). '
                             'Alias: --samples_per_class')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Alias for --samples_per_label (backward compat)')
    
    # Ratio mode
    parser.add_argument('--ratio', type=str, default=None,
                        help='Sample ratio per label, e.g. "4:2:1". '
                             'Must have same number of values as --labels_per_client.')
    parser.add_argument('--total_per_client', type=int, default=None,
                        help='Total samples per client in ratio mode. '
                             'Required when --ratio is used.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # ---- Handle backward-compatible aliases ----
    if args.classes_per_client is not None:
        args.labels_per_client = args.classes_per_client
    if args.samples_per_class is not None:
        args.samples_per_label = args.samples_per_class
    
    # ---- Build client specs ----
    if args.config:
        # JSON config mode — overrides all CLI flags
        print(f"\n  Loading config from: {args.config}")
        client_specs, seed = build_client_specs_from_config(args.config)
        print(f"  → {len(client_specs)} clients from {args.config}")
    else:
        # CLI mode — build uniform specs
        if args.ratio and not args.total_per_client:
            parser.error("--total_per_client is required when --ratio is used")
        
        client_specs = build_client_specs_from_args(args)
        seed = args.seed
        
        if args.ratio:
            print(f"\n  Ratio mode: {args.num_clients} clients × "
                  f"{args.labels_per_client} labels × ratio {args.ratio} "
                  f"× {args.total_per_client} total/client")
        else:
            print(f"\n  Uniform mode: {args.num_clients} clients × "
                  f"{args.labels_per_client} labels × "
                  f"{args.samples_per_label} samples/label")
    
    # ---- Run the generator ----
    generate_flexible_noniid(client_specs, seed=seed)
