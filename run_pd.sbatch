#!/bin/bash
#SBATCH --job-name=aeose
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:3
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name pd --actor-fc-dim 4 --timesteps 100000 --seed 99 --run-id 0 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name pd --actor-fc-dim 4 --timesteps 100000 --seed 99 --run-id 1 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name pd --actor-fc-dim 4 --timesteps 100000 --seed 99 --run-id 2 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name ppo --actor-fc-dim 4 --seed 99 & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --algo-name pd --actor-fc-dim 16 --timesteps 100000 --seed 99 --run-id 0 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --algo-name pd --actor-fc-dim 16 --timesteps 100000 --seed 99 --run-id 1 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --algo-name pd --actor-fc-dim 16 --timesteps 100000 --seed 99 --run-id 2 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --algo-name ppo --actor-fc-dim 16 --seed 99 & 

CUDA_VISIBLE_DEVICES=2 python3 main.py --algo-name pd --actor-fc-dim 32 --timesteps 100000 --seed 99 --run-id 0 & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --algo-name pd --actor-fc-dim 32 --timesteps 100000 --seed 99 --run-id 1 & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --algo-name pd --actor-fc-dim 32 --timesteps 100000 --seed 99 --run-id 2 & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --algo-name ppo --actor-fc-dim 32 --seed 99 & 

# === Wait for all background jobs to finish ===
wait