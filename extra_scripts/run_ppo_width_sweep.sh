#!/bin/bash

# Change to the root directory of the project
cd "$(dirname "$0")/.." || exit

# === GPU 0 ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 1 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 2 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 256 256 --seed 3 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64 --seed 1 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64 --seed 2 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 8 --seed 3 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4 --seed 1 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4 --seed 2 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 2 --seed 3 --env-name charge &

# === GPU 1 ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 1024 1024 --seed 2 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 1024 1024 --seed 3 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 256 256 --seed 1 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 64 64 --seed 2 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 64 64 --seed 3 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 8 --seed 1 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 4 --seed 2 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 4 --seed 3 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 1 --actor-fc-dim 2 --seed 1 --env-name resource &

# === GPU 2 ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 1024 1024 --seed 3 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 256 256 --seed 1 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 256 256 --seed 2 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 64 64 --seed 3 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 8 --seed 1 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 8 --seed 2 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 4 --seed 3 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 2 --seed 1 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 2 --actor-fc-dim 2 --seed 2 --env-name resource &

# === GPU 3 ===
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 1024 1024 --seed 1 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 256 256 --seed 2 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 256 256 --seed 3 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 64 64 --seed 1 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 8 --seed 2 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 8 --seed 3 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 4 --seed 1 --env-name resource &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 2 --seed 2 --env-name charge &
python3 main.py --project aeos --num-workers 3 --gpu-idx 3 --actor-fc-dim 2 --seed 3 --env-name resource &

wait
