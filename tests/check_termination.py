#!/usr/bin/env python3
"""Run short episodes and report when the environment signals termination.

Usage:
    python tests/check_termination.py --env downlink --episodes 3

The script prints the env's timing parameters up front, then per-step
term/trunc flags, sim_time (if available), and any info keys returned by
env.step(). It does not force a time-limit cap; it only reports what the env
itself does.
"""
import argparse
import sys
import os
import time

# Ensure repo root is on sys.path so local packages (e.g., `utils`) import
# correctly when the script is run from anywhere.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.get_env import get_env


def run_one_episode(env, verbose=True):
    obs = env.reset()
    step = 0
    while True:
        a = env.action_space.sample()
        ns, rew, term, trunc, info = env.step(a)
        step += 1

        sim_time = getattr(getattr(env, 'simulator', None), 'sim_time', None)
        if verbose:
            print(f"step={step:3d} term={term} trunc={trunc} rew={rew:.3f} sim_time={sim_time} info_keys={list(info.keys()) if isinstance(info, dict) else info}")

        # If env declares done, record and return
        if term or trunc:
            return dict(step=step, term=bool(term), trunc=bool(trunc), sim_time=sim_time, info=info)

        # safety
        if step > 10000:
            return dict(step=step, term=False, trunc=False, sim_time=sim_time, info=info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="downlink")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    try:
        env = get_env(args.env)
    except Exception as e:
        print("Failed to create env:", e)
        sys.exit(2)

    time_limit = getattr(env, 'time_limit', None)
    step_dur = getattr(env, 'max_step_duration', None)
    sim_rate = getattr(env, 'sim_rate', None)
    max_steps_attr = getattr(env, 'max_steps', None)

    print("env.time_limit", time_limit)
    print("env.max_step_duration", step_dur)
    print("env.sim_rate", sim_rate)
    print("env.max_steps (attr)", max_steps_attr)

    if time_limit is not None and step_dur:
        print("derived_decision_steps(time_limit/max_step_duration)", time_limit / step_dur)
        print("derived_decision_steps_int", int(time_limit / step_dur))

    if sim_rate:
        print("sim_dt_seconds(1/sim_rate)", 1.0 / sim_rate)
        if step_dur:
            print("sim_integration_steps_per_decision", step_dur * sim_rate)

    print("note: env.step() is one decision interval; the loop below stops only when env returns term or trunc")

    for ep in range(args.episodes):
        print('\n=== Episode', ep + 1, '===')
        res = run_one_episode(env)
        print('Result:', res)

    try:
        env.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
