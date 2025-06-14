import gymnasium as gym
import pytest

from bsk_rl import act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit


class TestImagingDynModelStorage:
    @pytest.mark.parametrize(
        "storage_capacity, initial_storage",
        [
            (8e7, -1),
            (8e7, 0),
            (8e7, 8e6),
            (10e7, 10e7),
            (8e7, 10e7),
        ],
    )
    def test_storageInit(self, storage_capacity, initial_storage):
        class ImageSat(sats.ImagingSatellite):
            dyn_type = dyn.ImagingDynModel
            fsw_type = fsw.ImagingFSWModel
            observation_spec = [obs.Time()]
            action_spec = [act.Downlink(), act.Image(n_ahead_image=10)]

        env = gym.make(
            "SatelliteTasking-v1",
            satellite=ImageSat(
                "EO-1",
                sat_args=ImageSat.default_sat_args(
                    oe=random_orbit,
                    dataStorageCapacity=storage_capacity,
                    storageInit=initial_storage,
                ),
            ),
            scenario=scene.UniformTargets(n_targets=1000),
            rewarder=data.NoReward(),
            sim_rate=1.0,
            time_limit=10000.0,
            max_step_duration=1e9,
            disable_env_checker=True,
        )

        env.reset()

        if initial_storage > storage_capacity or initial_storage < 0:
            assert env.unwrapped.satellite.dynamics.storage_level == 0
        else:
            assert env.unwrapped.satellite.dynamics.storage_level == initial_storage

    @pytest.mark.parametrize(
        "storage_capacity, initial_storage",
        [
            (8e7, 8e6),
            (10e7, 10e7),
        ],
    )
    def test_storageInit_downlink(self, storage_capacity, initial_storage):
        class ImageSat(sats.ImagingSatellite):
            dyn_type = dyn.FullFeaturedDynModel
            fsw_type = fsw.ImagingFSWModel
            observation_spec = [obs.Time()]
            action_spec = [act.Downlink()]

        env = gym.make(
            "SatelliteTasking-v1",
            satellite=ImageSat(
                "EO-1",
                sat_args=ImageSat.default_sat_args(
                    oe=random_orbit,
                    dataStorageCapacity=storage_capacity,
                    storageInit=initial_storage,
                ),
            ),
            scenario=scene.UniformTargets(n_targets=1000),
            rewarder=data.NoReward(),
            sim_rate=1.0,
            time_limit=10000.0,
            max_step_duration=1e9,
            disable_env_checker=True,
        )

        env.reset()

        terminated = False
        truncated = False
        while not terminated and not truncated:
            observation, reward, terminated, truncated, info = env.step(0)

        assert env.unwrapped.satellite.dynamics.storage_level < initial_storage


class TestConjunctionDynModel:
    @pytest.mark.parametrize(
        "rN1,vN1,collision",
        [
            ([1e8 + 30, 0, 0], [-1, 0, 0], True),
            ([1e8 + 30, 0, 0], [1, 0, 0], False),
            ([0, 1e8, 0], [0, 0, 0], False),
            ([1e8, 0, 0], [0, 0, 0], True),
        ],
    )
    def test_conjunction(self, rN1, vN1, collision):
        class CollisionSat(sats.Satellite):
            fsw_type = fsw.BasicFSWModel
            dyn_type = dyn.ConjunctionDynModel
            observation_spec = [obs.Time()]
            action_spec = [act.Drift()]

        env = gym.make(
            "ConstellationTasking-v1",
            satellites=[
                CollisionSat(
                    "Collision1",
                    sat_args=dict(
                        rN=rN1,
                        vN=vN1,
                        oe=None,
                    ),
                ),
                CollisionSat(
                    "Collision2",
                    sat_args=dict(
                        rN=[1e8, 0, 0],
                        vN=[0, 0, 0],
                        oe=None,
                    ),
                ),
                CollisionSat(
                    "NoCollision3",
                    sat_args=dict(
                        rN=[-1e8, 0, 0],
                        vN=[0, 0, 0],
                        oe=None,
                    ),
                ),
            ],
            sim_rate=1.0,
            time_limit=100.0,
            max_step_duration=1e9,
            disable_env_checker=True,
        )

        env.reset()

        env.step(dict(Collision1=0, Collision2=0, Collision3=0))

        sat1 = env.unwrapped.satellites[0]
        sat2 = env.unwrapped.satellites[1]
        sat3 = env.unwrapped.satellites[2]

        if collision:
            assert sat1.dynamics.conjunctions == [sat2]
            assert sat2.dynamics.conjunctions == [sat1]
            assert sat3.dynamics.conjunctions == []
        else:
            assert sat1.dynamics.conjunctions == []
            assert sat2.dynamics.conjunctions == []
            assert sat3.dynamics.conjunctions == []
