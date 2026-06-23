#!/bin/bash
#SBATCH --job-name=aeose_downlink
#SBATCH --account=bhqw-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=54  # 3 CPUs x 18 parallel python processes
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=aeose_downlink.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

source ~/.bashrc
conda activate aeos

ENV=downlink

# === All 18 runs on one A100 node, using GPU 0 only ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 3 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 3 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 16        --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 16        --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 16        --seed 3 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64     --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64     --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64     --seed 3 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 256 256   --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 256 256   --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 256 256   --seed 3 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 3 --env-name $ENV &

wait
