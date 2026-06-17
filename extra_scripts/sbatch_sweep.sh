#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=1-4          # Number of parallel agents (e.g. 4)
#SBATCH --gpus=1             # 1 GPU per agent (or adjust per your cluster specs)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# To run this script:
# 1. Create a logs directory: `mkdir -p logs`
# 2. Start a sweep from the login node first to get the SWEEP_ID:
#    `python search.py --count 0` (The --count 0 creates the sweep but runs no trials itself)
# 3. Replace 'your_sweep_id_here' below with the printed ID
# 4. Submit the job array: `sbatch sbatch_sweep.sh`

SWEEP_ID="your_sweep_id_here"
PROJECT="Exp"

# Optional: activate your environment
# source ~/.bashrc
# conda activate aeose

echo "Starting agent $SLURM_ARRAY_TASK_ID for sweep $SWEEP_ID"
python search.py --sweep_id $SWEEP_ID --count 10 --project $PROJECT
