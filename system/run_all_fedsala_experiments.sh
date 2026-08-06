#!/bin/bash
# ==============================================================================
# run_all_fedsala_experiments.sh — Master Experiment Runner for FedSALA Paper
# ==============================================================================
#
# Runs ALL 4 experiment groups automatically:
#
#   Group 1: CIFAR-10 Baseline Comparison
#            4 methods × 10 splits = 40 runs
#
#   Group 2: CIFAR-100 Baseline Comparison
#            4 methods × 3 splits = 12 runs
#
#   Group 3: FedSALA-L vs FedSALA (CIFAR-10)
#            2 methods × 10 splits = 20 runs
#
#   Group 4: FedSALA-G vs FedSALA (CIFAR-10)
#            2 methods × 10 splits = 20 runs
#
#   Total: 92 experiment runs
#
# USAGE:
#   cd FedSALA_github/system
#   chmod +x run_all_fedsala_experiments.sh
#   nohup ./run_all_fedsala_experiments.sh > ../results/fedsala_all_run.log 2>&1 &
#   tail -f ../results/fedsala_all_run.log
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ==============================================================================
# Master Results Directory
# ==============================================================================
RESULTS_DIR="../results/fedsala_all_experiments"
mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/run_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "  FEDSALA MASTER EXPERIMENT RUNNER"
echo "  Started:  $(date)"
echo "  Log:      $LOG_FILE"
echo "  Total:    92 experiment runs across 4 groups"
echo "================================================================"

# ==============================================================================
# Shared Hyperparameters
# ==============================================================================
ROUNDS=200
MODEL="resnet"
LR=0.001
LOCAL_STEPS=1
ETA=1.0
RAND_PERCENT=80

# CIFAR-10 specific
C10_BATCH=10
C10_THRESHOLD=0.7516455    # 75.1645% param coverage (matches layer_idx=17)
C10_NUM_CLASSES=10

# CIFAR-100 specific
C100_BATCH=16
C100_THRESHOLD=0.7526674   # 75.2667% param coverage (matches layer_idx=17)
C100_NUM_CLASSES=100

# Shared FedSALA/FedALA
EMA=0.5
SAMPLE=10
LAYER_IDX=17
# Set patience >= rounds to effectively disable early stopping
# (early stopping logic is also commented out in main.py)
PATIENCE=200

# ==============================================================================
# Split Definitions — Using Existing Scenario Names
# ==============================================================================
# Format: "SPLIT_ID:scenario_file.json:num_clients"

# CIFAR-10: 10 named splits
declare -a CIFAR10_SPLITS=(
    "S2:comp_2label.json:10"
    "S3:comp_3label.json:10"
    "S4:comp_4label.json:10"
    "SC:remote_sensing.json:10"
    "SD:hospital_uniform.json:10"
    "SE:hospital_mixed.json:10"
    "SF:camera_trap.json:15"
    "SA2:app_2label_skewed.json:15"
    "SA4:app_4label_skewed.json:5"
    "SMX:app_mixed_hetero.json:10"
)

# CIFAR-100: 3 named splits
declare -a CIFAR100_SPLITS=(
    "C100_5c:fedala_pathological_5c.json:20"
    "C100_10c:fedala_pathological_10c.json:20"
    "C100_20c:fedala_pathological_20c.json:20"
)

# ==============================================================================
# Counters
# ==============================================================================
TOTAL_RUNS=0
PASSED=0
FAILED=0
TOTAL_START=$SECONDS

# ==============================================================================
# Helper: Generate data for a split
# ==============================================================================
generate_data() {
    local DATASET_TYPE=$1     # "cifar10" or "cifar100"
    local SCENARIO=$2
    local SCENARIOS_DIR=$3

    echo "  [Data Gen] $SCENARIOS_DIR/$SCENARIO"
    if [ "$DATASET_TYPE" = "cifar10" ]; then
        $PYTHON generate_cifar10.py --config "$SCENARIOS_DIR/$SCENARIO"
    else
        $PYTHON generate_cifar100.py --config "$SCENARIOS_DIR/$SCENARIO"
    fi
}

# ==============================================================================
# Helper: Run a single experiment
# ==============================================================================
run_experiment() {
    local DATASET=$1          # "Cifar10" or "Cifar100"
    local NC=$2               # num clients
    local NB=$3               # num classes
    local BATCH=$4            # batch size
    local ALGO=$5             # algorithm name for -algo flag
    local GROUP_DIR=$6        # results subdirectory
    local RESULT_LABEL=$7     # label for result folder
    local EXTRA_ARGS=$8       # additional CLI args

    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    SECONDS=0

    echo ""
    echo "  >>> [$TOTAL_RUNS] ${RESULT_LABEL} ..."

    if $PYTHON main.py \
        -algo "$ALGO" \
        -gr $ROUNDS \
        -nc "$NC" \
        -nb "$NB" \
        -data "$DATASET" \
        -m $MODEL \
        -lbs "$BATCH" \
        -lr $LR \
        -ls $LOCAL_STEPS \
        --patience $PATIENCE \
        -dev cuda \
        $EXTRA_ARGS; then

        ELAPSED=$SECONDS
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))

        # Move result to organized folder
        LATEST=$(ls -td ../results/${DATASET}_* 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            TARGET="$GROUP_DIR/${RESULT_LABEL}"
            rm -rf "$TARGET"
            mv "$LATEST" "$TARGET"
            echo "${ELAPSED_MIN}m ${ELAPSED_SEC}s" > "$TARGET/training_time.txt"
            echo "    ✓ Done in ${ELAPSED_MIN}m${ELAPSED_SEC}s → $TARGET"
        fi
        PASSED=$((PASSED + 1))
    else
        echo "    ✗ FAILED: ${RESULT_LABEL}"
        FAILED=$((FAILED + 1))
    fi
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GROUP 1: CIFAR-10 BASELINE COMPARISON                                  ║
# ║  Methods: LocalOnly, FedAvg, FedALA, FedSALA                           ║
# ║  Splits: 10 CIFAR-10 scenarios                                          ║
# ║  Total: 40 runs                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

GROUP1_DIR="$RESULTS_DIR/group1_cifar10_baseline"
mkdir -p "$GROUP1_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GROUP 1: CIFAR-10 BASELINE COMPARISON (40 runs)           ║"
echo "║  Methods: LocalOnly | FedAvg | FedALA | FedSALA            ║"
echo "║  Started: $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"

for SPLIT_ENTRY in "${CIFAR10_SPLITS[@]}"; do
    IFS=':' read -r SPLIT_ID SCENARIO NC <<< "$SPLIT_ENTRY"

    echo ""
    echo "  ════════════════════════════════════════"
    echo "  Split: ${SPLIT_ID} (${SCENARIO}, ${NC} clients)"
    echo "  ════════════════════════════════════════"

    generate_data "cifar10" "$SCENARIO" "scenarios"

    # --- LocalOnly ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "LocalOnly" "$GROUP1_DIR" "${SPLIT_ID}_LocalOnly" ""

    # --- FedAvg ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedAvg" "$GROUP1_DIR" "${SPLIT_ID}_FedAvg" ""

    # --- FedALA ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedALA" "$GROUP1_DIR" "${SPLIT_ID}_FedALA" \
        "-p $LAYER_IDX -s $RAND_PERCENT -et $ETA"

    # --- FedSALA (Method 3) ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedSALA" "$GROUP1_DIR" "${SPLIT_ID}_FedSALA" \
        "--fedsala_method 3 --fisher_threshold $C10_THRESHOLD --fisher_ema_alpha $EMA --fisher_sample_percent $SAMPLE"

    # --- Generate comparison plots for this split ---
    echo "  [Comparison] Generating combined plots for ${SPLIT_ID}..."
    $PYTHON compare_split_results.py \
        --split_id "$SPLIT_ID" \
        --group_dir "$GROUP1_DIR" \
        --methods LocalOnly FedAvg FedALA FedSALA
done

echo ""
echo "  ✓ GROUP 1 COMPLETE ($(date))"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GROUP 2: CIFAR-100 BASELINE COMPARISON                                 ║
# ║  Methods: LocalOnly, FedAvg, FedALA, FedSALA                           ║
# ║  Splits: 3 CIFAR-100 scenarios                                          ║
# ║  Total: 12 runs                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

GROUP2_DIR="$RESULTS_DIR/group2_cifar100_baseline"
mkdir -p "$GROUP2_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GROUP 2: CIFAR-100 BASELINE COMPARISON (12 runs)          ║"
echo "║  Methods: LocalOnly | FedAvg | FedALA | FedSALA            ║"
echo "║  Started: $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"

for SPLIT_ENTRY in "${CIFAR100_SPLITS[@]}"; do
    IFS=':' read -r SPLIT_ID SCENARIO NC <<< "$SPLIT_ENTRY"

    echo ""
    echo "  ════════════════════════════════════════"
    echo "  Split: ${SPLIT_ID} (${SCENARIO}, ${NC} clients)"
    echo "  ════════════════════════════════════════"

    generate_data "cifar100" "$SCENARIO" "scenarios_cifar100"

    # --- LocalOnly ---
    run_experiment "Cifar100" "$NC" "$C100_NUM_CLASSES" "$C100_BATCH" \
        "LocalOnly" "$GROUP2_DIR" "${SPLIT_ID}_LocalOnly" ""

    # --- FedAvg ---
    run_experiment "Cifar100" "$NC" "$C100_NUM_CLASSES" "$C100_BATCH" \
        "FedAvg" "$GROUP2_DIR" "${SPLIT_ID}_FedAvg" ""

    # --- FedALA ---
    run_experiment "Cifar100" "$NC" "$C100_NUM_CLASSES" "$C100_BATCH" \
        "FedALA" "$GROUP2_DIR" "${SPLIT_ID}_FedALA" \
        "-p $LAYER_IDX -s $RAND_PERCENT -et $ETA"

    # --- FedSALA (Method 3) ---
    run_experiment "Cifar100" "$NC" "$C100_NUM_CLASSES" "$C100_BATCH" \
        "FedSALA" "$GROUP2_DIR" "${SPLIT_ID}_FedSALA" \
        "--fedsala_method 3 --fisher_threshold $C100_THRESHOLD --fisher_ema_alpha $EMA --fisher_sample_percent $SAMPLE"

    # --- Generate comparison plots for this split ---
    echo "  [Comparison] Generating combined plots for ${SPLIT_ID}..."
    $PYTHON compare_split_results.py \
        --split_id "$SPLIT_ID" \
        --group_dir "$GROUP2_DIR" \
        --methods LocalOnly FedAvg FedALA FedSALA
done

echo ""
echo "  ✓ GROUP 2 COMPLETE ($(date))"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GROUP 3: FedALA-L vs FedSALA (CIFAR-10)                                ║
# ║  FedALA-L  = FedALA with --lower_layers_local                           ║
# ║  FedSALA   = FedSALA Method 3 (High-ALA, Low-Local)                     ║
# ║  Splits: 10 CIFAR-10 scenarios                                          ║
# ║  Total: 20 runs                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

GROUP3_DIR="$RESULTS_DIR/group3_fedala_l_comparison"
mkdir -p "$GROUP3_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GROUP 3: FedALA-L vs FedSALA (20 runs)                     ║"
echo "║  FedALA-L  = FedALA + lower_layers_local                    ║"
echo "║  FedSALA   = FedSALA Method 3                               ║"
echo "║  Started: $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"

for SPLIT_ENTRY in "${CIFAR10_SPLITS[@]}"; do
    IFS=':' read -r SPLIT_ID SCENARIO NC <<< "$SPLIT_ENTRY"

    echo ""
    echo "  ════════════════════════════════════════"
    echo "  Split: ${SPLIT_ID} (${SCENARIO}, ${NC} clients)"
    echo "  ════════════════════════════════════════"

    generate_data "cifar10" "$SCENARIO" "scenarios"

    # --- FedALA-L (FedALA with lower_layers_local) ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedALA" "$GROUP3_DIR" "${SPLIT_ID}_FedALA-L" \
        "-p $LAYER_IDX -s $RAND_PERCENT -et $ETA --lower_layers_local"

    # --- FedSALA (Method 3) ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedSALA" "$GROUP3_DIR" "${SPLIT_ID}_FedSALA" \
        "--fedsala_method 3 --fisher_threshold $C10_THRESHOLD --fisher_ema_alpha $EMA --fisher_sample_percent $SAMPLE"

    # --- Generate comparison plots for this split ---
    # algo_names maps display labels to .npy file prefixes:
    #   FedALA-L uses FedALA internally, FedSALA uses FedSALA
    echo "  [Comparison] Generating combined plots for ${SPLIT_ID}..."
    $PYTHON compare_split_results.py \
        --split_id "$SPLIT_ID" \
        --group_dir "$GROUP3_DIR" \
        --methods FedALA-L FedSALA \
        --algo_names FedALA FedSALA
done

echo ""
echo "  ✓ GROUP 3 COMPLETE ($(date))"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GROUP 4: FedSALA-G vs FedSALA (CIFAR-10)                              ║
# ║  FedSALA-G = FedSALA Method 1 (High-ALA, Low-Global)                   ║
# ║  FedSALA   = FedSALA Method 3 (High-ALA, Low-Local)                    ║
# ║  Splits: 10 CIFAR-10 scenarios                                          ║
# ║  Total: 20 runs                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

GROUP4_DIR="$RESULTS_DIR/group4_fedsala_g_comparison"
mkdir -p "$GROUP4_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GROUP 4: FedSALA-G vs FedSALA (20 runs)                   ║"
echo "║  FedSALA-G = Method 1 (High-ALA, Low-Global)                ║"
echo "║  FedSALA   = Method 3 (High-ALA, Low-Local)                 ║"
echo "║  Started: $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"

for SPLIT_ENTRY in "${CIFAR10_SPLITS[@]}"; do
    IFS=':' read -r SPLIT_ID SCENARIO NC <<< "$SPLIT_ENTRY"

    echo ""
    echo "  ════════════════════════════════════════"
    echo "  Split: ${SPLIT_ID} (${SCENARIO}, ${NC} clients)"
    echo "  ════════════════════════════════════════"

    generate_data "cifar10" "$SCENARIO" "scenarios"

    # --- FedSALA-G (Method 1: High-ALA, Low-Global) ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedSALA" "$GROUP4_DIR" "${SPLIT_ID}_FedSALA-G" \
        "--fedsala_method 1 --fisher_threshold $C10_THRESHOLD --fisher_ema_alpha $EMA --fisher_sample_percent $SAMPLE"

    # --- FedSALA (Method 3: High-ALA, Low-Local) ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedSALA" "$GROUP4_DIR" "${SPLIT_ID}_FedSALA" \
        "--fedsala_method 3 --fisher_threshold $C10_THRESHOLD --fisher_ema_alpha $EMA --fisher_sample_percent $SAMPLE"

    # --- Generate comparison plots for this split ---
    # algo_names maps display labels to .npy file prefixes:
    #   FedSALA-G uses FedSALA internally (method 1), FedSALA uses FedSALA (method 3)
    echo "  [Comparison] Generating combined plots for ${SPLIT_ID}..."
    $PYTHON compare_split_results.py \
        --split_id "$SPLIT_ID" \
        --group_dir "$GROUP4_DIR" \
        --methods FedSALA-G FedSALA \
        --algo_names FedSALA FedSALA
done

echo ""
echo "  ✓ GROUP 4 COMPLETE ($(date))"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
TOTAL_HR=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================"
echo "  ALL FEDSALA EXPERIMENTS FINISHED"
echo "  Completed: $(date)"
echo "  Total time: ${TOTAL_HR}h ${TOTAL_MIN}m"
echo "  Runs: $TOTAL_RUNS | Passed: $PASSED | Failed: $FAILED"
echo ""
echo "  Results:"
echo "    Group 1 (CIFAR-10 Baseline):    $GROUP1_DIR/"
echo "    Group 2 (CIFAR-100 Baseline):   $GROUP2_DIR/"
echo "    Group 3 (FedALA-L vs FedSALA):  $GROUP3_DIR/"
echo "    Group 4 (FedSALA-G vs FedSALA): $GROUP4_DIR/"
echo "  Log: $LOG_FILE"
echo "================================================================"
