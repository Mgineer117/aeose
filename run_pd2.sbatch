#!/bin/bash
#SBATCH --job-name=aeose
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 98 --run-id 0 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 99 --run-id 0 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 100 --run-id 0 & 

CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 98 --run-id 1 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 99 --run-id 1 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 100 --run-id 1 & 

CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 98 --run-id 2 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 99 --run-id 2 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 2 --timesteps 10000000 --seed 100 --run-id 2 &

CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 2 --seed 98 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 2 --seed 99 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 2 --seed 100 & 



# === Wait for all background jobs to finish ===
wait