# AEOSE

Reinforcement learning experiments for spacecraft tasks (charge, desat, downlink, resource) built on top of [bsk_rl](https://github.com/AVSLab/bsk_rl).

## Installation

### 0. Create a Miniconda environment with Python 3.11

This project is developed and tested with Python 3.11.* in a Miniconda environment.

```bash
conda create -n aeos python==3.11.*
conda activate aeos
```

### 1. Install dependencies

From the project root, install Basilisk, `bsk_rl`, and the Python requirements in one command:

```bash
pip install bsk bsk-rl -r requirements.txt
```

## Running

Train an agent with `main.py`. Pick an environment and an algorithm:

```bash
python main.py --env-name charge --algo-name ppo
```

### Common arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--env-name` | `charge` | Environment: `charge`, `desat`, `downlink`, `resource` |
| `--algo-name` | `ppo` | Algorithm: `ppo`, `ddpg`, `pd` |
| `--seed` | `42` | Random seed |
| `--num-runs` | `5` | Number of seeds to run |
| `--timesteps` | `1.2e7` | Total training timesteps |
| `--project` | `Exp` | WandB project name |
| `--logdir` | `log/train_log` | Output log directory |

Run `python main.py --help` for the full list of options.

### Examples

```bash
# PPO on the desat environment
python main.py --env-name desat --algo-name ppo

# DDPG on resource with a custom seed
python main.py --env-name resource --algo-name ddpg --seed 0

# PD trainer on downlink
python main.py --env-name downlink --algo-name pd
```

### Cluster jobs

SLURM batch scripts are provided under [commands/sbatch/](commands/sbatch/). Submit one with:

```bash
sbatch commands/sbatch/csl/run_aeose_charge_base1.sbatch
```

## Project layout

- [main.py](main.py) — training entry point
- [algorithms/](algorithms/) — PPO, DDPG, PD implementations
- [envs/](envs/) — task environments (charge, desat, downlink, resource)
- [policy/](policy/) — policy networks
- [utils/](utils/) — args, env factory, replay buffer, samplers
- [requirements.txt](requirements.txt) — Python dependencies
