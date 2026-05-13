#!/bin/bash
#SBATCH --job-name=aeose_charge
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=aeose_charge.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

source ~/.bashrc
conda activate aeos

ENV=charge

# === All actor-fc-dim configs × seeds on a single GPU ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1         --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1         --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1         --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 2         --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 2         --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 2         --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 4         --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 4         --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 4         --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 64 64     --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 64 64     --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 64 64     --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 256 256   --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 256 256   --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 256 256   --seed 3 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1024 1024 --seed 1 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1024 1024 --seed 2 --env-name $ENV &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1024 1024 --seed 3 --env-name $ENV &

wait
