#!/bin/bash
#SBATCH --job-name=resource
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --output=aeose.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Load conda    
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate aeose

# === Run Experiments in Parallel === #
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 1 --seed 1 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 1 --seed 2 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 1 --seed 3 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 2 --seed 1 --env-name resource & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 2 --seed 2 --env-name resource & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 2 --seed 3 --env-name resource & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 4 --seed 1 --env-name resource & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 4 --seed 2 --env-name resource & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeose --actor-fc-dim 4 --seed 3 --env-name resource & 

# === Wait for all background jobs to finish ===
wait