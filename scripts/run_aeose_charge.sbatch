#!/bin/bash
#SBATCH --job-name=aeose_charge
#SBATCH --account=bhqw-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=aeose_charge.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

source ~/.bashrc
conda activate aeos

export WANDB_INIT_TIMEOUT=300       # extend WandB init timeout to 5 min for parallel launches

ENV=charge

# === 9 runs on GPU 0 (small architectures) ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 3 --env-name $ENV &
sleep 10
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 3 --env-name $ENV &
sleep 10
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 16        --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 16        --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 16        --seed 3 --env-name $ENV &
sleep 10

# === 9 runs on GPU 1 (large architectures) ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 64 64     --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 64 64     --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 64 64     --seed 3 --env-name $ENV &
sleep 10
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 256 256   --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 256 256   --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 256 256   --seed 3 --env-name $ENV &
sleep 10
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 1024 1024 --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 1024 1024 --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 1024 1024 --seed 3 --env-name $ENV &

wait
