#!/bin/bash
#SBATCH --job-name=aeose_desat
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=aeose_desat.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

source ~/.bashrc
conda activate aeos

ENV=desat

# === All actor-fc-dim configs × seeds on a single GPU ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 1         --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 1         --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 1         --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 2         --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 2         --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 2         --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 4         --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 4         --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 4         --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 64 64     --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 64 64     --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 64 64     --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 256 256   --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 256 256   --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 256 256   --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 1024 1024 --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 1024 1024 --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 3 --actor-fc-dim 1024 1024 --seed 3 --env-name $ENV &

wait
