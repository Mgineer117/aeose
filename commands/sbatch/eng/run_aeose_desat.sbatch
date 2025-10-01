#!/bin/bash
#SBATCH --job-name=aeose
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:3
#SBATCH --time=2-00:00:00
#SBATCH --output=aeose.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Load conda    
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate aeose

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 64 64 --seed 1 --env-name desat & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 64 64 --seed 2 --env-name desat & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 64 64 --seed 3 --env-name desat & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 256 256 --seed 1 --env-name desat & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 256 256 --seed 2 --env-name desat & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 256 256 --seed 3 --env-name desat & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeose --actor-fc-dim 1024 1024 --seed 1 --env-name desat & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeose --actor-fc-dim 1024 1024 --seed 2 --env-name desat & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeose --actor-fc-dim 1024 1024 --seed 3 --env-name desat &

# === Wait for all background jobs to finish ===
wait