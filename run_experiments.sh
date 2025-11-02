#!/bin/bash
#
# Convenience script to run experiments with different configurations
#

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mnt/ssd_ext/huggingface/envs/qwen_prunevid

# Set the number of samples for experiments
# Use a small number for testing, increase for full evaluation
NUM_SAMPLES=50  # Change to omit --num_samples for full dataset

echo "========================================="
echo "Running Qwen2-VL + PruneVid Experiments"
echo "========================================="
echo "Samples per experiment: $NUM_SAMPLES"
echo ""

# Experiment 1: Baseline (no pruning)
echo "[1/5] Running baseline evaluation (no pruning)..."
python eval_egoschema_qwen_prunevid.py \
    --no_pruning \
    --num_samples $NUM_SAMPLES \
    --exp_name baseline_${NUM_SAMPLES}samples

echo ""
echo "Baseline complete!"
echo ""

# Experiment 2: Light pruning (keep 80%)
echo "[2/5] Running light pruning (keep_ratio=0.8)..."
python eval_egoschema_qwen_prunevid.py \
    --enable_pruning \
    --keep_ratio 0.8 \
    --num_samples $NUM_SAMPLES \
    --exp_name pruned_ratio0.8_${NUM_SAMPLES}samples

echo ""
echo "Light pruning complete!"
echo ""

# Experiment 3: Medium pruning (keep 60%)
echo "[3/5] Running medium pruning (keep_ratio=0.6)..."
python eval_egoschema_qwen_prunevid.py \
    --enable_pruning \
    --keep_ratio 0.6 \
    --num_samples $NUM_SAMPLES \
    --exp_name pruned_ratio0.6_${NUM_SAMPLES}samples

echo ""
echo "Medium pruning complete!"
echo ""

# Experiment 4: Aggressive pruning (keep 40%)
echo "[4/5] Running aggressive pruning (keep_ratio=0.4)..."
python eval_egoschema_qwen_prunevid.py \
    --enable_pruning \
    --keep_ratio 0.4 \
    --num_samples $NUM_SAMPLES \
    --exp_name pruned_ratio0.4_${NUM_SAMPLES}samples

echo ""
echo "Aggressive pruning complete!"
echo ""

# Experiment 5: Very aggressive pruning (keep 30%)
echo "[5/5] Running very aggressive pruning (keep_ratio=0.3)..."
python eval_egoschema_qwen_prunevid.py \
    --enable_pruning \
    --keep_ratio 0.3 \
    --num_samples $NUM_SAMPLES \
    --exp_name pruned_ratio0.3_${NUM_SAMPLES}samples

echo ""
echo "Very aggressive pruning complete!"
echo ""

# Generate comparison report
echo "========================================="
echo "All experiments complete!"
echo "========================================="
echo ""
echo "Results saved to: results/"
echo ""
echo "Experiment summaries:"
for exp in baseline_${NUM_SAMPLES}samples pruned_ratio0.8_${NUM_SAMPLES}samples pruned_ratio0.6_${NUM_SAMPLES}samples pruned_ratio0.4_${NUM_SAMPLES}samples pruned_ratio0.3_${NUM_SAMPLES}samples; do
    if [ -f "results/$exp/summary.json" ]; then
        echo ""
        echo "--- $exp ---"
        python -c "
import json
with open('results/$exp/summary.json') as f:
    s = json.load(f)
    print(f\"  Accuracy: {s.get('accuracy', 0):.2%}\")
    print(f\"  Avg Time: {s.get('avg_inference_time', 0):.2f}s\")
    if 'keep_ratio' in s:
        print(f\"  Keep Ratio: {s['keep_ratio']}\")
"
    fi
done

echo ""
echo "Done!"
