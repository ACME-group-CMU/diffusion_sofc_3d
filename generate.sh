#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 
#SBATCH --cpus-per-task=12
#SBATCH --mem=1T
#SBATCH --time=2-00:00:00
#SBATCH --output ./Outputs/%j.out
#SBATCH --error ./Outputs/%j.err
#SBATCH --mail-user=rbajpai@andrew.cmu.edu
#SBATCH --mail-type=ALL

module load aocc/4.1.0 cuda/12.4
module load anaconda/3
source activate rbenv

cd ~/conditional_diffusion/diffusion_sofc_3d

# --- Directory and Paths ---
mkdir -p ./generated_samples

# --- Configuration Variables ---
# Edit these variables to customize your generation run

# Model and checkpoint settings
BASE_PATH="/.nfs/home/6/bajpair/conditional_diffusion/diffusion_sofc_3d/results/lightning_logs"

if [ -n "$1" ] && [ -n "$2" ]; then
    VERSION="$1"
    USE_EMA="$2"
    echo "Using provided version: $VERSION and EMA setting: $USE_EMA"
else
    VERSION="13"
    USE_EMA="true"
    echo "Using default version: $VERSION and EMA setting: $USE_EMA"
fi

# Validate EMA setting
if [ "$USE_EMA" != "true" ] && [ "$USE_EMA" != "false" ]; then
    echo "Error: EMA setting must be 'true' or 'false', got: $USE_EMA"
    exit 1
fi

NAME="version_$VERSION"  # Name for this generation run
CHECKPOINT_PATH=""  # Set this for specific checkpoint path
CHECKPOINT_TYPE="ALL"  # best_val, best_train, or last (used with VERSION)

# Output settings
OUTPUT_PATH="./generated_samples/run_$NAME.npz"

# Generation parameters
NUM_SAMPLES=96              # Total samples across all GPUs
BATCH_SIZE_PER_GPU=12        # Samples per batch per GPU
IMG_SIZE=96                  # Image size (cubic)
CHANNELS=1                   # Number of channels
INF_TIMESTEPS=1000          # Inference timesteps

# Data source settings
USE_RANDOM=false              # true=random samples (pure noise), false=validation data (noisy samples)
DATA_PATH="/nfs/home/6/bajpair/conditional_diffusion/datasets/subvolumes/" # Path to data directory (required when USE_RANDOM=false)
CHARACTERISTICS=()  # Condition characteristics (empty array for unconditional)

# Guidance settings
W_GUIDANCE=-1.0              # Guidance weight (-1.0 disables guidance)

# EMA settings
#USE_EMA=true               # Set to true to use EMA weights

# Hardware settings
GPUS_PYTHON_ARG=8           # Number of GPUs to use
NUM_WORKERS=4               # Number of data loading workers

# --- Auto-configuration ---
# Automatically determine checkpoint path if not explicitly set
if [ -n "$VERSION" ] && [ -z "$CHECKPOINT_PATH" ]; then
    BASE_DIR="./results/lightning_logs/version_${VERSION}/checkpoints"
    
    case $CHECKPOINT_TYPE in
        "best_val")
            # Find checkpoint with lowest validation loss
            CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_val_loss-*.ckpt" -type f | \
                while read -r file; do
                    # Extract loss value from filename (e.g., best_val_loss-epoch=071-val_loss=0.025649.ckpt)
                    loss=$(echo "$file" | sed -n 's/.*val_loss=\([0-9]*\.[0-9]*\)\.ckpt/\1/p')
                    if [ -n "$loss" ]; then
                        echo "$loss $file"
                    fi
                done | sort -n | head -1 | cut -d' ' -f2-)
            
            # Fallback if no loss value found in filename
            if [ -z "$CHECKPOINT_PATH" ]; then
                CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_val_loss-*.ckpt" -type f | head -1)
            fi
            ;;
        "best_train")
            # Find checkpoint with lowest training loss
            CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_loss-*.ckpt" -type f | \
                while read -r file; do
                    # Extract loss value from filename (e.g., best_loss-epoch=043-loss=0.029324.ckpt)
                    loss=$(echo "$file" | sed -n 's/.*loss=\([0-9]*\.[0-9]*\)\.ckpt/\1/p')
                    if [ -n "$loss" ]; then
                        echo "$loss $file"
                    fi
                done | sort -n | head -1 | cut -d' ' -f2-)
            
            # Fallback if no loss value found in filename
            if [ -z "$CHECKPOINT_PATH" ]; then
                CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_loss-*.ckpt" -type f | head -1)
            fi
            ;;
        "last")
            CHECKPOINT_PATH="$BASE_DIR/last.ckpt"
            ;;
        "ALL")
            # Find ALL checkpoints but exclude best_val, best_train, last, and files without both epoch AND step
            CHECKPOINT_LIST=$(find "$BASE_DIR" -name "*.ckpt" -type f | \
                while read -r file; do
                    filename=$(basename "$file")
                    
                    # Skip best_val, best_train, and last checkpoints
                    if [[ "$filename" == best_val_loss-* ]] || \
                       [[ "$filename" == best_loss-* ]] || \
                       [[ "$filename" == "last.ckpt" ]]; then
                        continue
                    fi
                    
                    # Only include files with BOTH "epoch" AND "step" in the name
                    if [[ "$filename" == *"epoch"* ]] && [[ "$filename" == *"step"* ]]; then
                        # Extract epoch number from filename for sorting
                        epoch=$(echo "$file" | sed -n 's/.*epoch=\([0-9]*\).*/\1/p')
                        if [ -n "$epoch" ]; then
                            # Convert to integer for proper numerical sorting
                            printf "%05d %s\n" "$epoch" "$file"
                        fi
                    fi
                done | sort -k1,1 | cut -d' ' -f2-)
            
            if [ -z "$CHECKPOINT_LIST" ]; then
                echo "Error: No checkpoints found for ALL in $BASE_DIR"
                exit 1
            fi
            ;;
    esac
    
    if [ "$CHECKPOINT_TYPE" != "ALL" ]; then
        if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "Error: Could not find $CHECKPOINT_TYPE checkpoint in $BASE_DIR"
            echo "Available checkpoints:"
            ls -la "$BASE_DIR" 2>/dev/null || echo "Directory not found"
            exit 1
        fi
    fi
fi

# --- Build command arguments ---
DATA_ARGS=""
if [ "$USE_RANDOM" = true ]; then
    DATA_ARGS="--use_random"
    echo "Using random sample generation (pure noise)"
else
    if [ ! -d "$DATA_PATH" ]; then
        echo "Error: Data path not found: $DATA_PATH"
        exit 1
    fi
    DATA_ARGS="--data_path $DATA_PATH"
    echo "Using validation data from: $DATA_PATH (noisy samples)"
fi

# Build characteristics arguments
CHAR_ARGS=""
if [ ${#CHARACTERISTICS[@]} -gt 0 ]; then
    CHAR_ARGS="--characteristics ${CHARACTERISTICS[*]}"
fi

USE_EMA_ARGS=""
if [ "$USE_EMA" = true ]; then
    USE_EMA_ARGS="--use_ema"
fi

# --- Execution Summary ---
echo "=" * 60
echo "🚀 GENERATION SUMMARY"
echo "=" * 60
echo "📂 Checkpoint: $CHECKPOINT_PATH"
echo "💾 Output: $OUTPUT_PATH"
echo "📊 Samples: $NUM_SAMPLES"
if [ "$USE_RANDOM" = true ]; then
    echo "🎲 Data source: Random samples (pure noise)"
else
    echo "📁 Data source: Validation data (noisy samples from $DATA_PATH)"
fi
if [ ${#CHARACTERISTICS[@]} -gt 0 ]; then
    echo "🎯 Conditions: ${CHARACTERISTICS[*]}"
else
    echo "🎯 Mode: Unconditional"
fi
echo "🎨 Image size: ${IMG_SIZE}³, Channels: $CHANNELS"
echo "🎨 Inference timesteps: $INF_TIMESTEPS, Guidance: $W_GUIDANCE"
echo "🎨 Using EMA: $USE_EMA"
echo "🖥️  GPUs: $GPUS_PYTHON_ARG, Workers: $NUM_WORKERS"
echo "🖥️  Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "=" * 60

echo "Python script will be launched with --gpus $GPUS_PYTHON_ARG and --num_nodes $SLURM_NNODES"
echo "Total samples: $NUM_SAMPLES, Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Expected samples per GPU: $(($NUM_SAMPLES / $GPUS_PYTHON_ARG))"

# --- Execution ---
echo ""
echo "🎬 Starting generation..."

if [ "$CHECKPOINT_TYPE" = "ALL" ]; then
    OUTDIR="./generated_samples/version_${VERSION}_ema_${USE_EMA}_${NUM_SAMPLES}samples"
    mkdir -p "$OUTDIR"
    
    # Count total checkpoints for progress tracking
    TOTAL_CKPTS=$(echo "$CHECKPOINT_LIST" | wc -l)
    CURRENT_CKPT=0
    
    echo "Found $TOTAL_CKPTS checkpoints to process (sorted by epoch number):"
    echo "$CHECKPOINT_LIST" | while read -r ckpt; do
        epoch=$(echo "$ckpt" | sed -n 's/.*epoch=\([0-9]*\).*/\1/p')
        if [ -n "$epoch" ]; then
            echo "  - $(basename "$ckpt") (epoch $epoch)"
        else
            echo "  - $(basename "$ckpt") (no epoch found)"
        fi
    done
    echo ""
    
    for CKPT in $CHECKPOINT_LIST; do
        CKPT_BASENAME=$(basename "$CKPT" .ckpt)
        OUTFILE="${OUTDIR}/run_${NAME}_${CKPT_BASENAME}.npz"
        echo ""
        echo "🎬 Generating from checkpoint: $CKPT"
        srun python3 inference.py \
            --version "${VERSION}" \
            --checkpoint_path "${CKPT}" \
            --output_path "${OUTFILE}" \
            --num_samples ${NUM_SAMPLES} \
            ${DATA_ARGS} \
            ${CHAR_ARGS} \
            --batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
            --img_size ${IMG_SIZE} \
            --channels ${CHANNELS} \
            --inf_timesteps ${INF_TIMESTEPS} \
            --w ${W_GUIDANCE} \
            ${USE_EMA_ARGS} \
            --gpus ${GPUS_PYTHON_ARG} \
            --num_nodes ${SLURM_NNODES} \
            --num_workers ${NUM_WORKERS}
        if [ $? -eq 0 ]; then
            echo "✅ Generated samples for $CKPT_BASENAME saved to $OUTFILE"
        else
            echo "❌ Generation failed for $CKPT_BASENAME"
        fi
    done
    echo ""
    echo "🎉 ALL checkpoint generations completed!"
    exit 0
fi

srun python3 inference.py \
    --version "${VERSION}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --num_samples ${NUM_SAMPLES} \
    ${DATA_ARGS} \
    ${CHAR_ARGS} \
    --batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
    --img_size ${IMG_SIZE} \
    --channels ${CHANNELS} \
    --inf_timesteps ${INF_TIMESTEPS} \
    --w ${W_GUIDANCE} \
    ${USE_EMA_ARGS} \
    --gpus ${GPUS_PYTHON_ARG} \
    --num_nodes ${SLURM_NNODES} \
    --num_workers ${NUM_WORKERS}

# --- Post-execution ---
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Inference finished successfully!"
    echo "📊 Samples saved to: ${OUTPUT_PATH}"
    
    # Show file info if available
    if [ -f "$OUTPUT_PATH" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "📏 File size: $FILE_SIZE"
        
        # Try to show basic info about the generated file
        if command -v python >/dev/null 2>&1; then
            echo "📋 File contents:"
            python -c "
import numpy as np
try:
    data = np.load('$OUTPUT_PATH')
    for key in data.keys():
        item = data[key]
        if hasattr(item, 'shape'):
            print(f'  {key}: shape {item.shape}, dtype {item.dtype}')
        else:
            print(f'  {key}: {type(item).__name__}')
except Exception as e:
    print(f'  Could not read file info: {e}')
" 2>/dev/null
        fi
    fi
    
    echo ""
    echo "🎉 Generation completed successfully!"
else
    echo ""
    echo "❌ Generation failed!"
    echo "Check the SLURM output files for details:"
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "  Output: ./Outputs/${SLURM_JOB_ID}.out"
        echo "  Error:  ./Outputs/${SLURM_JOB_ID}.err"
    fi
    exit 1
fi
