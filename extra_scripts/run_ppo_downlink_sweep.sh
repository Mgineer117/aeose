#!/bin/bash

# Change to the root directory of the project
cd "$(dirname "$0")/.." || exit

# === GPU 0: downlink ===
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 2 --seed 1 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 2 --seed 2 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 2 --seed 3 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4 --seed 1 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4 --seed 2 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 4 --seed 3 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 8 --seed 1 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 8 --seed 2 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 8 --seed 3 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64 --seed 1 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64 --seed 2 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 64 64 --seed 3 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 256 256 --seed 1 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 256 256 --seed 2 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 256 256 --seed 3 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 1 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 2 --env-name downlink &
nohup python3 main.py --project aeos --num-workers 3 --gpu-idx 0 --actor-fc-dim 1024 1024 --seed 3 --env-name downlink &

wait
