# ga_agile_eos.py
# Genetic Algorithm baseline for Agile EOS (single-sensor) exactly per paper operators/probabilities.
# - Representation: fixed-length action sequence over decision intervals (open-loop plan)
# - Fitness: sum of per-step rewards from the environment (identical to RL)
# - Operators: one-point crossover p=0.25; uniform mutation p_seq=0.25, p_gene=0.30; tournament selection size=3
# - Action alphabet: {Charge, Desaturate, Downlink, Image c0, Image c1, Image c2}
# - Imaging indices c0..c2 map to the env’s current subset U of up to 3 nearest unimaged targets at each step.

from __future__ import annotations
import argparse
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
import gymnasium as gym

# DEAP
from deap import base, creator, tools

# Action encoding (genes)
# We keep genes as ints for speed: 0..5
# 0=CHARGE, 1=DESAT, 2=DOWNLINK, 3=IMAGE_c0, 4=IMAGE_c1, 5=IMAGE_c2
GENE_MEANINGS = [
    "CHARGE", "DESAT", "DOWNLINK", "IMAGE_c0", "IMAGE_c1", "IMAGE_c2"
]
GENE_MIN, GENE_MAX = 0, 5

def gene_to_str(g: int) -> str:
    return GENE_MEANINGS[g]

# Environment adapter
@dataclass
class EnvInfo:
    horizon: int
    n_actions: int

class AgileEOSAdapter:
    """
    A thin adapter so the GA can remain agnostic to env internals:
      - figures out the decision horizon
      - maps {CHARGE, DESAT, DOWNLINK, IMAGE_c0..c2} to env action integers at each step
      - rolls out one full episode from a fixed action sequence and returns total reward
    """
    def __init__(self, env_id: str, seed: Optional[int] = None, env_kwargs: Optional[Dict[str, Any]] = None):
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        self.rng = np.random.default_rng(seed)
        self.env = gym.make(env_id, **self.env_kwargs)
        if seed is not None:
            try:
                self.env.reset(seed=seed)
            except TypeError:
                # older gym versions use env.seed
                if hasattr(self.env, "seed"):
                    self.env.seed(seed)
        self.info = self._inspect_env()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    #  Introspection 
    def _inspect_env(self) -> EnvInfo:
        # Determine horizon (|I|). Use (in order of preference):
        #  1) env.unwrapped.max_intervals or decision_intervals (common in scheduling envs)
        #  2) env.spec.max_episode_steps
        #  3) probe by stepping a no-op until done (fallback; capped)
        horizon = None
        unwrapped = getattr(self.env, "unwrapped", self.env)

        for attr in ["max_intervals", "decision_intervals", "horizon", "episode_len"]:
            if hasattr(unwrapped, attr) and isinstance(getattr(unwrapped, attr), int):
                horizon = int(getattr(unwrapped, attr))
                break

        if horizon is None:
            if getattr(self.env.spec, "max_episode_steps", None):
                horizon = int(self.env.spec.max_episode_steps)

        if horizon is None:
            # Fallback probe (safe default cap)
            obs, _ = self.env.reset()
            horizon = 0
            cap = 1000
            for _ in range(cap):
                a = self._fallback_safe_action(obs)
                obs, r, terminated, truncated, _ = self.env.step(a)
                horizon += 1
                if terminated or truncated:
                    break
            if horizon >= cap:
                print("[WARN] Could not infer horizon confidently; using 300.", file=sys.stderr)
                horizon = 300

        # Number of primitive env actions (not the gene alphabet)
        if hasattr(self.env.action_space, "n"):
            n_actions = int(self.env.action_space.n)
        else:
            raise RuntimeError("Env action space must be Discrete.")

        return EnvInfo(horizon=horizon, n_actions=n_actions)

    def _fallback_safe_action(self, obs: Any) -> int:
        """
        If we cannot map an imaging gene (due to no candidates), fall back to a conservative action.
        Prefer CHARGE if available; else choose action 0.
        """
        # Try to find CHARGE meaning from env if exposed, else just return 0.
        if hasattr(self.env, "action_meanings"):
            meanings = self.env.action_meanings
            if isinstance(meanings, (list, tuple)):
                for i, m in enumerate(meanings):
                    if str(m).upper().startswith("CHARGE"):
                        return i
        return 0

    # --- Gene -> Env action (per-step) ---
    def _map_gene_to_action(self, obs: Any, gene: int) -> int:
        """
        Map our 6-gene alphabet to the env’s Discrete action set at this time step.
        We assume the agile EOS env exposes either:
          - a stable ordering where 0:Charge, 1:Desat, 2:Downlink, and imaging actions are specific indices, OR
          - a helper on 'unwrapped' to build an imaging action from subset index k (0..2), OR
          - the observation encodes a small action mask or U-subset indices we can translate via a method.

        This function tries common patterns used in bsk_rl environments:
          1) env.unwrapped.action_from_mode(mode, k=None)
          2) env.unwrapped.imaging_action(k) / get_imaging_action(k)
          3) env.action_meanings to locate base modes
        If imaging c_k isn't available at this step (no target in that slot), we gracefully fall back to CHARGE.
        """
        unwrapped = getattr(self.env, "unwrapped", self.env)

        # Non-imaging modes (CHARGE, DESAT, DOWNLINK) first:
        if gene in (0, 1, 2):
            mode_name = {0: "CHARGE", 1: "DESAT", 2: "DOWNLINK"}[gene]
            # Try helper to resolve by name:
            if hasattr(unwrapped, "action_from_mode"):
                try:
                    return int(unwrapped.action_from_mode(mode_name))
                except Exception:
                    pass
            # Otherwise, try action_meanings lookup:
            if hasattr(self.env, "action_meanings"):
                meanings = self.env.action_meanings
                for i, m in enumerate(meanings):
                    if str(m).upper().startswith(mode_name):
                        return i
            # Last resort:
            return self._fallback_safe_action(obs)

        # Imaging genes: IMAGE_c0..c2
        k = gene - 3  # 0,1,2

        # Preferred helpers:
        for helper in ["imaging_action", "get_imaging_action", "action_from_mode"]:
            if hasattr(unwrapped, helper):
                try:
                    if helper == "action_from_mode":
                        a = unwrapped.action_from_mode("IMAGE", k=k)
                    else:
                        a = getattr(unwrapped, helper)(k)
                    return int(a)
                except Exception:
                    pass

        # If the obs contains a mask or subset, try to resolve. Common patterns:
        # - obs may be a dict with "subset" or "U" giving valid target slots.
        # - some envs expose a per-step list of valid actions; we can guess that IMAGE slots map to consistent indices.
        # If not resolvable, fall back:
        return self._fallback_safe_action(obs)

    # --- Rollout one full episode from a fixed action sequence ---
    def evaluate_sequence(self, genes: List[int], eval_seed: Optional[int] = None) -> float:
        total_r = 0.0
        # Reset with optional seed for deterministic evaluation
        if eval_seed is not None:
            obs, _ = self.env.reset(seed=eval_seed)
        else:
            obs, _ = self.env.reset()

        steps = min(len(genes), self.info.horizon)

        for t in range(steps):
            a = self._map_gene_to_action(obs, genes[t])
            obs, r, terminated, truncated, _ = self.env.step(a)
            total_r += float(r)
            if terminated or truncated:
                break

        return total_r

# GA setup (DEAP)
def make_deap_toolbox(seq_len: int,
                      cxpb: float = 0.25,
                      mutpb: float = 0.25,
                      indpb: float = 0.30,
                      tournsize: int = 3,
                      rng_seed: Optional[int] = None):
    """
    Create DEAP toolbox per paper:
      - one-point crossover with p=0.25
      - uniform mutation: mutate a whole sequence with p=0.25, and each gene with p=0.30
      - tournament selection size=3
    """
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    rnd = random.Random(rng_seed)

    # Gene initializer: randint in [0,5]
    toolbox.register("gene", rnd.randint, GENE_MIN, GENE_MAX)

    # Individual = list of genes of length seq_len
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=seq_len)

    # Population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Crossover: one point
    toolbox.register("mate", tools.cxOnePoint)

    # Mutation: for each gene, with prob indpb, sample a new gene uniformly in [0,5] (could also mutate by resampling != current)
    def mut_uniform_individual(individual: List[int]) -> Tuple[List[int],]:
        for i in range(len(individual)):
            if rnd.random() < indpb:
                # resample gene (allow equal; simpler and unbiased)
                individual[i] = rnd.randint(GENE_MIN, GENE_MAX)
        return (individual,)

    toolbox.register("mutate", mut_uniform_individual)

    # Selection: tournament of size 3
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    # Store operator probabilities on the toolbox for later reference
    toolbox.cxpb = cxpb
    toolbox.mutpb = mutpb
    toolbox.indpb = indpb
    toolbox.tournsize = tournsize
    toolbox._rng = rnd
    return toolbox

# End-to-end GA run
def run_ga(env_id: str,
           env_kwargs: Optional[Dict[str, Any]],
           population_size: int,
           generations: int,
           horizon: Optional[int],
           seed: Optional[int],
           eval_seed: Optional[int],
           verbose: bool = True) -> Dict[str, Any]:
    # Build env adapter
    adapter = AgileEOSAdapter(env_id=env_id, seed=seed, env_kwargs=env_kwargs)
    seq_len = horizon if horizon is not None else adapter.info.horizon

    toolbox = make_deap_toolbox(seq_len=seq_len, rng_seed=seed)

    # Fitness function (identical to env return)
    def eval_individual(individual: List[int]) -> Tuple[float]:
        # Sequence-level mutation probability (p=0.25) is applied by DEAP's evolve loop via toolbox.mutate with mutpb;
        # per-gene probability (0.30) is inside our mutate() above.
        total_return = adapter.evaluate_sequence(individual, eval_seed=eval_seed)
        return (total_return,)

    toolbox.register("evaluate", eval_individual)

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Trackers
    best = tools.HallOfFame(1)
    best.update(pop)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log = []
    if verbose:
        print(f"[GA] seq_len={seq_len}, pop={population_size}, gens={generations}, "
              f"cxpb={toolbox.cxpb}, mutpb={toolbox.mutpb}, indpb={toolbox.indpb}, tourn={toolbox.tournsize}")

    # Evolutionary loop (μ+λ style via select over parents+offspring)
    for gen in range(1, generations + 1):
        offspring = tools.selTournament(pop, len(pop), tournsize=toolbox.tournsize)

        # Clone
        offspring = list(map(toolbox.clone, offspring))

        # Crossover (applied pairwise) with cxpb=0.25
        for i in range(1, len(offspring), 2): 
            if toolbox._rng.random() < toolbox.cxpb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # Mutation at sequence-level with mutpb=0.25
        for i, ind in enumerate(offspring):
            if toolbox._rng.random() < toolbox.mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Evaluate invalid fitness
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        # Combine and select next gen
        pop = toolbox.select(pop + offspring, k=population_size)

        # Update trackers
        best.update(pop)
        record = stats.compile(pop)
        record["gen"] = gen
        log.append(record)
        if verbose and (gen == 1 or gen % 10 == 0 or gen == generations):
            print(f"[GA] gen={gen:4d}  avg={record['avg']:.4f}  std={record['std']:.4f}  "
                  f"min={record['min']:.4f}  max={record['max']:.4f}")

    # Final best rollout (optional, for confirmation)
    best_ind = best[0]
    best_return = adapter.evaluate_sequence(best_ind, eval_seed=eval_seed)

    if verbose:
        print("\n[GA] Best individual (gene strings):")
        print(" ".join(gene_to_str(g) for g in best_ind))
        print(f"[GA] Best fitness (re-eval): {best_return:.6f}")

    adapter.close()
    return {
        "best_individual": list(best_ind),
        "best_return": float(best_return),
        "log": log,
        "seq_len": seq_len
    }

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="GA baseline for Agile EOS with DEAP (paper-accurate operators).")
    p.add_argument("--env", type=str, default="bsk_rl.envs.agile_eos:AgileEOSEnv",
                   help="Gymnasium env ID or entry-point path, e.g., bsk_rl.envs.agile_eos:AgileEOSEnv")
    p.add_argument("--pop", type=int, default=200, help="Population size")
    p.add_argument("--gens", type=int, default=200, help="Generations")
    p.add_argument("--horizon", type=int, default=None,
                   help="Override sequence length (|I|). Defaults to env horizon if omitted.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for GA & env construction")
    p.add_argument("--eval_seed", type=int, default=123, help="Seed for deterministic fitness evaluation")
    p.add_argument("--no_verbose", action="store_true", help="Silence per-10-gen logging")
    # You can push env JSON through here later if needed
    return p.parse_args()

def main():
    args = parse_args()
    res = run_ga(
        env_id=args.env,
        env_kwargs={},
        population_size=args.pop,
        generations=args.gens,
        horizon=args.horizon,
        seed=args.seed,
        eval_seed=args.eval_seed,
        verbose=not args.no_verbose
    )
    # Minimal summary:
    print("\n=== GA Summary ===")
    print(f"Best return: {res['best_return']:.6f}")
    print(f"Seq length:  {res['seq_len']}")
    print("First 32 genes:", " ".join(gene_to_str(g) for g in res["best_individual"][:32]))

if __name__ == "__main__":
    main()
