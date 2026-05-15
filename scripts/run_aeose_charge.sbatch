#!/bin/bash
#SBATCH --job-name=aeose_charge
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=7
#SBATCH --mem=440G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=3-00:00:00
#SBATCH --output=aeose_charge.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

source ~/.bashrc
conda activate aeos

ENV=charge

# === All 18 runs on one A100 node, single GPU ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1         --seed 3 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 2         --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 2         --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 2         --seed 3 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 1 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 2 --env-name $ENV &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4         --seed 3 --env-name $ENV &
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
