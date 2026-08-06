#!/bin/bash
# ==============================================================================
# run_fedsala_75_vs_100.sh — FedSALA 75.16% vs FedSALA 100% (CIFAR-10)
# ==============================================================================
#
# Compares FedSALA with 75.16% high-Fisher ALA zone (default)
# against FedSALA with 100% ALA zone (all parameters get ALA blending).
#
#   FedSALA-75  = Method 3, fisher_threshold=0.7516455 (top 75.16% → ALA)
#   FedSALA-100 = Method 3, fisher_threshold=1.0       (top 100%   → ALA)
#
#   Both use the same EMA (0.5), sampling (10%), and all other hyperparameters.
#
#   Splits: 10 CIFAR-10 scenarios
#   Total:  20 runs (2 methods × 10 splits)
#
# USAGE:
#   cd FedALA/system
#   chmod +x run_fedsala_75_vs_100.sh
#   nohup ./run_fedsala_75_vs_100.sh > ../results/fedsala_75_vs_100_run.log 2>&1 &
#   tail -f ../results/fedsala_75_vs_100_run.log
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ==============================================================================
# Results Directory
# ==============================================================================
RESULTS_DIR="../results/fedsala_75_vs_100"
mkdir -p "$RESULTS_DIR"

LOG_FILE="$RESULTS_DIR/run_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "  FedSALA 75.16% vs FedSALA 100% — CIFAR-10"
echo "  Started:  $(date)"
echo "  Log:      $LOG_FILE"
echo "  Total:    20 experiment runs (2 methods × 10 splits)"
echo "================================================================"

# ==============================================================================
# Shared Hyperparameters
# ==============================================================================
ROUNDS=200
MODEL="resnet"
LR=0.001
LOCAL_STEPS=1
C10_BATCH=10
C10_NUM_CLASSES=10

# FedSALA shared
EMA=0.5
SAMPLE=10
# Set patience >= rounds to effectively disable early stopping
PATIENCE=200

# The two configurations being compared
THRESHOLD_75=0.7516455    # 75.16% param coverage (matches layer_idx=17)
THRESHOLD_100=1.0         # 100% param coverage (all params in ALA zone)

# ==============================================================================
# Split Definitions — Same 10 CIFAR-10 scenarios as other groups
# ==============================================================================
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

# ==============================================================================
# Counters
# ==============================================================================
TOTAL_RUNS=0
PASSED=0
FAILED=0
TOTAL_START=$SECONDS

# ==============================================================================
# Helper: Run a single experiment
# ==============================================================================
run_experiment() {
    local DATASET=$1
    local NC=$2
    local NB=$3
    local BATCH=$4
    local ALGO=$5
    local GROUP_DIR=$6
    local RESULT_LABEL=$7
    local EXTRA_ARGS=$8

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

        # Move result to organized folder (DO NOT delete existing results)
        LATEST=$(ls -td ../results/${DATASET}_* 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            TARGET="$GROUP_DIR/${RESULT_LABEL}"
            # Safety: if target already exists, back it up instead of deleting
            if [ -d "$TARGET" ]; then
                BACKUP="${TARGET}_backup_${TIMESTAMP}"
                echo "    ⚠ Target exists, backing up to ${BACKUP}"
                mv "$TARGET" "$BACKUP"
            fi
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
# ║  FedSALA 75.16% vs FedSALA 100% (CIFAR-10)                             ║
# ║  FedSALA-75  = Method 3, threshold=0.7516455 (75.16% ALA zone)         ║
# ║  FedSALA-100 = Method 3, threshold=1.0       (100% ALA zone)           ║
# ║  Splits: 10 CIFAR-10 scenarios                                          ║
# ║  Total: 20 runs                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FedSALA 75.16% vs FedSALA 100% (20 runs)                  ║"
echo "║  FedSALA-75  = Method 3, threshold=0.7516455                ║"
echo "║  FedSALA-100 = Method 3, threshold=1.0 (all params → ALA)  ║"
echo "║  Started: $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"

for SPLIT_ENTRY in "${CIFAR10_SPLITS[@]}"; do
    IFS=':' read -r SPLIT_ID SCENARIO NC <<< "$SPLIT_ENTRY"

    echo ""
    echo "  ════════════════════════════════════════"
    echo "  Split: ${SPLIT_ID} (${SCENARIO}, ${NC} clients)"
    echo "  ════════════════════════════════════════"

    # Generate data for this split
    echo "  [Data Gen] scenarios/$SCENARIO"
    $PYTHON generate_cifar10.py --config "scenarios/$SCENARIO"

    # --- FedSALA 75.16% (default threshold) ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedSALA" "$RESULTS_DIR" "${SPLIT_ID}_FedSALA-75" \
        "--fedsala_method 3 --fisher_threshold $THRESHOLD_75 --fisher_ema_alpha $EMA --fisher_sample_percent $SAMPLE"

    # --- FedSALA 100% (all params in ALA zone) ---
    run_experiment "Cifar10" "$NC" "$C10_NUM_CLASSES" "$C10_BATCH" \
        "FedSALA" "$RESULTS_DIR" "${SPLIT_ID}_FedSALA-100" \
        "--fedsala_method 3 --fisher_threshold $THRESHOLD_100 --fisher_ema_alpha $EMA --fisher_sample_percent $SAMPLE"

    # --- Generate comparison plots for this split ---
    echo "  [Comparison] Generating combined plots for ${SPLIT_ID}..."
    $PYTHON compare_split_results.py \
        --split_id "$SPLIT_ID" \
        --group_dir "$RESULTS_DIR" \
        --methods FedSALA-75 FedSALA-100 \
        --algo_names FedSALA FedSALA
done

echo ""
echo "  ✓ ALL RUNS COMPLETE ($(date))"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
TOTAL_HR=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "================================================================"
echo "  FedSALA 75.16% vs FedSALA 100% — FINISHED"
echo "  Completed: $(date)"
echo "  Total time: ${TOTAL_HR}h ${TOTAL_MIN}m"
echo "  Runs: $TOTAL_RUNS | Passed: $PASSED | Failed: $FAILED"
echo ""
echo "  Results: $RESULTS_DIR/"
echo "  Log:     $LOG_FILE"
echo "================================================================"
