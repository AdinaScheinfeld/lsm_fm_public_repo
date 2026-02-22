#!/bin/bash
#SBATCH --job-name=pretrain_unet
#SBATCH --output=../output/logs/pretrain_unet_%j.out
#SBATCH --error=../output/logs/pretrain_unet_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu <-- update to your cluster's GPU partition name if different
#SBATCH --gres=gpu:2 # select number of gpus ***
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # 1 task per gpu ***


# pretrain_unet_job.sh - Script to run pretraining

# =============================================================================
# USER CONFIGURATION - update these before running
# =============================================================================
# 1. Update --partition to your cluster's GPU partition name (above)
# 2. Update the anaconda module name to match your cluster (module load anaconda3/...) (below)
# 3. Set your wandb API key: export WANDB_API_KEY=your_key_here before submitting (below)
# 4. Adjust --gres, --mem, --cpus-per-task, --ntasks-per-node to match your hardware (above)
# =============================================================================

# indicate starting
echo "Starting LSM pretraining..."



# activate conda environment
module load anaconda3  # <-- update to your cluster's module name for Anaconda if different
source activate lsm-pretrain

# wandb login <-- see instructions below
# set your wandb API key before running: Uncomment the line below and replace with your key or run `wandb login` interactively before submitting this job
# export WANDB_API_KEY=your_key_here
if [ -z "$WANDB_API_KEY" ]; then
    echo "[WARNING] WANDB_API_KEY not set. Run 'wandb login' or set the variable before submitting."
fi

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0 # when set to 0, allows NCCL to use P2P communication for better performance (instead of using CPU for communication)
export NCCL_IB_DISABLE=0 # when set to 0, allows NCCL to use InfiniBand for better performance (instead of using TCP/IP for communication)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # when set to 1, allows NCCL to handle errors asynchronously (can improve performance in some cases)

# Hugging Face offline / cache config
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=120

# create output directories if they don't exist
mkdir -p ../output/logs

# set path to config file
CONFIG_PATH="pretrain_config_unet.yaml"

# clock job start time
export START_EPOCH="$(date +%s)"
echo "[INFO] Job runtime timer started at $(date -d @${START_EPOCH} '+%Y-%m-%d %H:%M:%S')"

# allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run script
srun --label python pretrain_unet.py --config $CONFIG_PATH



# indicate completion
echo 'LSM pretraining complete'













