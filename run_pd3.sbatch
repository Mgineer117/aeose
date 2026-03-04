#!/bin/bash
#SBATCH --job-name=aeose
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 98 --run-id 0  & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 99 --run-id 0  & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 100 --run-id 0  & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 101 --run-id 0  & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 102 --run-id 0  & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 98 --run-id 1  & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 99 --run-id 1  & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 100 --run-id 1  & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 101 --run-id 1  & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 102 --run-id 1  & 

CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 98 --run-id 2  & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 99 --run-id 2  & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 100 --run-id 2  &
CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 101 --run-id 2  &
CUDA_VISIBLE_DEVICES=2 python3 main.py --project aeos_pd --env-name desat --algo-name pd --actor-fc-dim 4 --timesteps 5000000 --seed 102 --run-id 2  &

CUDA_VISIBLE_DEVICES=3 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 4 --seed 98 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 4 --seed 99 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 4 --seed 100 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 4 --seed 101 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --project aeos_pd --env-name desat --algo-name ppo --actor-fc-dim 4 --seed 102 & 

# === Wait for all background jobs to finish ===
wait