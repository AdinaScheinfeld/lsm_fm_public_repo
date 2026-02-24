#!/bin/bash
#SBATCH --job-name=finetune_unet
#SBATCH --output=../output/logs/finetune_unet_%j.out
#SBATCH --error=../output/logs/finetune_unet_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu   # <-- update to your cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# finetune_segmentation_unet_job.sh - SLURM script for UNet finetuning

# =============================================================================
# USER CONFIGURATION - update these before running
# =============================================================================
# 1. Update --partition to your cluster's GPU partition name
# 2. Update the anaconda module name to match your cluster
# 3. Set your wandb API key: export WANDB_API_KEY=your_key_here before submitting
#    or run `wandb login` interactively before submitting
# 4. Set CONFIG_PATH to point to your config yaml file
# =============================================================================

echo "Starting LSM UNet finetuning..."

# activate conda environment
module load anaconda3        # <-- update to match your cluster's module name
source activate lsm-pretrain

# wandb login <-- see instructions below
# set your wandb API key before running: Uncomment the line below and replace with your key or run `wandb login` interactively before submitting this job
# export WANDB_API_KEY=your_key_here
if [ -z "$WANDB_API_KEY" ]; then
    echo "[WARNING] WANDB_API_KEY not set. Run 'wandb login' or set the variable before submitting."
fi

# environment variables
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# create output log directory
mkdir -p ../output/logs

# path to config file — update this to point to your config
CONFIG_PATH="finetune_segmentation_unet_config.yaml"

# clock job start time
export START_EPOCH="$(date +%s)"
echo "[INFO] Job started at $(date -d @${START_EPOCH} '+%Y-%m-%d %H:%M:%S')"

# run finetuning
srun --label python finetune_segmentation_unet.py --config "$CONFIG_PATH"

# log end time
END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "[INFO] Job finished at $(date -d @${END_EPOCH} '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Total runtime: $((ELAPSED / 3600)):$(( (ELAPSED % 3600) / 60 )):$(( ELAPSED % 60 ))s"

echo "LSM UNet finetuning complete."



