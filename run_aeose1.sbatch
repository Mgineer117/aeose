#!/bin/bash
#SBATCH --job-name=aeose
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=aeose.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Load conda    
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate aeose

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 8 8 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 16 16 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 32 32 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 64 64 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 128 128 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 256 256 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 512 512 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 1024 1024 & 
# CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 1536 1536 & 
# CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 2048 2048 & 
# CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 4096 4096 & 
# CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 8192 8192 & 

# === Wait for all background jobs to finish ===
wait