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

# --- Parse Arguments ---
VERSION=${VERSION:-${1:-"9"}}
CHECKPOINT_TYPE=${CHECKPOINT_TYPE:-${2:-"ALL"}}
NUM_SAMPLES=${NUM_SAMPLES:-${3:-96}}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-${4:-12}}
USE_EMA=${USE_EMA:-${5:-"true"}}

# Optional parameters via environment variables (unchanged)
CONDITION_FILE=${CONDITION_FILE:-""}
NOISE_FILE=${NOISE_FILE:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"./generated_samples/Clip_Accumulate_TimeWeight/version_${VERSION}_${NUM_SAMPLES}samples_ema_${USE_EMA}/"}
INF_TIMESTEPS=${INF_TIMESTEPS:-1000}
W_GUIDANCE=${W_GUIDANCE:-0.0}
GPUS=${GPUS:-8}
NUM_WORKERS=${NUM_WORKERS:-4}

# --- Validation ---
if [ "$USE_EMA" != "true" ] && [ "$USE_EMA" != "false" ]; then
   echo "Error: USE_EMA must be 'true' or 'false', got: $USE_EMA"
   exit 1
fi

if ! [[ "$CHECKPOINT_TYPE" =~ ^(best_val|best_train|last|ALL)$ ]]; then
   echo "Error: CHECKPOINT_TYPE must be one of: best_val, best_train, last, ALL"
   exit 1
fi

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

# --- Checkpoint Discovery ---
BASE_DIR="./results/lightning_logs/version_${VERSION}/checkpoints"

if [ ! -d "$BASE_DIR" ]; then
   echo "Error: Checkpoint directory not found: $BASE_DIR"
   exit 1
fi

case $CHECKPOINT_TYPE in
   "best_val")
       CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_val_loss-*.ckpt" -type f | \
           while read -r file; do
               loss=$(echo "$file" | sed -n 's/.*val_loss=\([0-9]*\.[0-9]*\)\.ckpt/\1/p')
               if [ -n "$loss" ]; then
                   echo "$loss $file"
               fi
           done | sort -n | head -1 | cut -d' ' -f2-)
       
       if [ -z "$CHECKPOINT_PATH" ]; then
           CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_val_loss-*.ckpt" -type f | head -1)
       fi
       ;;
   "best_train")
       CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_loss-*.ckpt" -type f | \
           while read -r file; do
               loss=$(echo "$file" | sed -n 's/.*loss=\([0-9]*\.[0-9]*\)\.ckpt/\1/p')
               if [ -n "$loss" ]; then
                   echo "$loss $file"
               fi
           done | sort -n | head -1 | cut -d' ' -f2-)
       
       if [ -z "$CHECKPOINT_PATH" ]; then
           CHECKPOINT_PATH=$(find "$BASE_DIR" -name "best_loss-*.ckpt" -type f | head -1)
       fi
       ;;
   "last")
       CHECKPOINT_PATH="$BASE_DIR/last.ckpt"
       ;;
   "ALL")
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
                   epoch=$(echo "$file" | sed -n 's/.*epoch=\([0-9]*\).*/\1/p')
                   if [ -n "$epoch" ]; then
                       printf "%05d %s\n" $((10#$epoch)) "$file"
                   fi
               fi
           done | sort -k1,1 | cut -d' ' -f2-)
       
       if [ -z "$CHECKPOINT_LIST" ]; then
           echo "Error: No periodic checkpoints found in $BASE_DIR"
           exit 1
       fi
       ;;
esac

# Validate single checkpoint exists
if [ "$CHECKPOINT_TYPE" != "ALL" ]; then
   if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
       echo "Error: Could not find $CHECKPOINT_TYPE checkpoint in $BASE_DIR"
       exit 1
   fi
fi

# --- Build inference arguments ---
INFERENCE_ARGS="--num_samples $NUM_SAMPLES"
INFERENCE_ARGS="$INFERENCE_ARGS --batch_size_per_gpu $BATCH_SIZE_PER_GPU"
INFERENCE_ARGS="$INFERENCE_ARGS --inf_timesteps $INF_TIMESTEPS"
INFERENCE_ARGS="$INFERENCE_ARGS --w $W_GUIDANCE"
INFERENCE_ARGS="$INFERENCE_ARGS --gpus $GPUS"
INFERENCE_ARGS="$INFERENCE_ARGS --num_nodes $SLURM_NNODES"
INFERENCE_ARGS="$INFERENCE_ARGS --num_workers $NUM_WORKERS"

if [ "$USE_EMA" = "true" ]; then
   INFERENCE_ARGS="$INFERENCE_ARGS --use_ema"
fi

if [ -n "$CONDITION_FILE" ] && [ -f "$CONDITION_FILE" ]; then
   INFERENCE_ARGS="$INFERENCE_ARGS --condition_file $CONDITION_FILE"
fi

if [ -n "$NOISE_FILE" ] && [ -f "$NOISE_FILE" ]; then
   INFERENCE_ARGS="$INFERENCE_ARGS --noise_file $NOISE_FILE"
fi

# --- Execution Summary ---
echo "=================================="
echo "🚀 GENERATION SETUP"
echo "=================================="
echo "📂 Version: $VERSION"
echo "🎯 Checkpoint type: $CHECKPOINT_TYPE"
echo "📊 Samples: $NUM_SAMPLES"
echo "🖥️  GPUs: $GPUS, Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "🎨 EMA: $USE_EMA, Timesteps: $INF_TIMESTEPS"
if [ -n "$CONDITION_FILE" ]; then
   echo "🎯 Conditions: $CONDITION_FILE"
fi
if [ -n "$NOISE_FILE" ]; then
   echo "🎲 Noise: $NOISE_FILE"
fi
echo "💾 Output: $OUTPUT_DIR"
echo "=================================="

# --- Execute ---
if [ "$CHECKPOINT_TYPE" = "ALL" ]; then
   TOTAL_CKPTS=$(echo "$CHECKPOINT_LIST" | wc -l)
   CURRENT_CKPT=0
   SUCCESS_CKPTS=0
   
   echo "🎬 Processing $TOTAL_CKPTS checkpoints..."
   echo ""
   
   for CKPT in $CHECKPOINT_LIST; do
       CURRENT_CKPT=$((CURRENT_CKPT + 1))
       CKPT_BASENAME=$(basename "$CKPT" .ckpt)
       OUTPUT_PATH="${OUTPUT_DIR}/version_${VERSION}_${CKPT_BASENAME}.npz"
       
       echo "[$CURRENT_CKPT/$TOTAL_CKPTS] Processing: $(basename "$CKPT")"
       
       if [ -f "$OUTPUT_PATH" ]; then
           echo "⏭️  Output exists, skipping: $(basename "$OUTPUT_PATH")"
           SUCCESS_CKPTS=$((SUCCESS_CKPTS + 1))
           continue
       fi
       
       srun python3 inference.py \
           --version "$VERSION" \
           --checkpoint_path "$CKPT" \
           --output_path "$OUTPUT_PATH" \
           $INFERENCE_ARGS
       
       if [ $? -eq 0 ] && [ -f "$OUTPUT_PATH" ]; then
           echo "✅ Success: $(basename "$OUTPUT_PATH")"
           SUCCESS_CKPTS=$((SUCCESS_CKPTS + 1))
       else
           echo "❌ Failed: $(basename "$CKPT")"
       fi
       echo ""
   done
   
   echo "🎉 Completed: $SUCCESS_CKPTS/$TOTAL_CKPTS successful"
else
   # Single checkpoint
   CKPT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)
   OUTPUT_PATH="${OUTPUT_DIR}/version_${VERSION}_${CKPT_NAME}.npz"
   
   echo "🎬 Processing: $(basename "$CHECKPOINT_PATH")"
   
   srun python3 inference.py \
       --version "$VERSION" \
       --checkpoint_path "$CHECKPOINT_PATH" \
       --output_path "$OUTPUT_PATH" \
       $INFERENCE_ARGS
   
   if [ $? -eq 0 ]; then
       echo "✅ Generation completed: $OUTPUT_PATH"
   else
       echo "❌ Generation failed"
       exit 1
   fi
fi
