#!/bin/bash
#SBATCH --job-name=aeose_downlink
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=aeose_downlink.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

source ~/.bashrc
conda activate aeose

ENV=downlink
# downlink originally trained for 2.1e7 timesteps; preserve that.
TS=21000000

# === All actor-fc-dim configs × seeds on a single GPU ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1         --seed 1 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1         --seed 2 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1         --seed 3 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 2         --seed 1 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 2         --seed 2 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 2         --seed 3 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 4         --seed 1 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 4         --seed 2 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 4         --seed 3 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 64 64     --seed 1 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 64 64     --seed 2 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 64 64     --seed 3 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 256 256   --seed 1 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 256 256   --seed 2 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 256 256   --seed 3 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1024 1024 --seed 1 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1024 1024 --seed 2 --env-name $ENV --timesteps $TS &
CUDA_VISIBLE_DEVICES=0 python3 main.py --project aeos --num-workers 4 --actor-fc-dim 1024 1024 --seed 3 --env-name $ENV --timesteps $TS &

wait
