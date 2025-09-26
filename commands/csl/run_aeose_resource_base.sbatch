#!/bin/bash
#SBATCH --job-name=aeose
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=aeose.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Load conda    
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate aeose

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 2 --seed 1 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 2 --seed 2 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 2 --seed 3 --env-name resource & 

CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 8 --seed 1 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 8 --seed 2 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 8 --seed 3 --env-name resource & 

CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 32 --seed 1 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 32 --seed 2 --env-name resource & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeose --actor-fc-dim 32 --seed 3 --env-name resource &   


# === Wait for all background jobs to finish ===
wait