from copy import deepcopy

import numpy as np
from Basilisk.utilities import orbitalMotion
from bsk_rl.scene.targets import Target, UniformTargets
from bsk_rl.utils.orbital import random_orbit

duration = 5700.0 * 5  # 5 orbits, matching Herrmann 2023
decision_interval = 360.0  # 6 min planning interval
orbit_alt_km = 800 # 500
target_distribution = "uniform"
n_targets = 135
n_ahead = 3


def _paper_priority_distribution():
	return float(np.random.choice([1, 2, 3]))


class GroundTrackTargets(UniformTargets):
	"""Approximate the paper's along-ground-track target generation.

	The paper samples targets along the spacecraft ground track and perturbs
	their positions slightly off-nadir. We approximate that here by sampling
	positions along a single reference orbit and projecting them onto the
	Earth surface with a small lateral angular offset.
	"""

	def __init__(
		self,
		n_targets,
		alt_km: float = orbit_alt_km,
		max_offset_deg: float = 2.0,
		priority_distribution=None,
		radius: float = orbitalMotion.REQ_EARTH * 1e3,
	):
		super().__init__(
			n_targets=n_targets,
			priority_distribution=priority_distribution or _paper_priority_distribution,
			radius=radius,
		)
		self.alt_km = alt_km
		self.max_offset_deg = max_offset_deg

	def regenerate_targets(self) -> None:
		self.targets = []
		base_oe = random_orbit(alt=self.alt_km)
		anomalies = np.linspace(0.0, 2.0 * np.pi, self.n_targets, endpoint=False)
		anomalies += np.random.uniform(
			low=-np.pi / max(1, self.n_targets),
			high=np.pi / max(1, self.n_targets),
			size=self.n_targets,
		)
		anomalies = np.sort(np.mod(anomalies, 2.0 * np.pi))

		for i, anomaly in enumerate(anomalies):
			oe = deepcopy(base_oe)
			oe.f = anomaly
			r_orbit, _ = orbitalMotion.elem2rv(orbitalMotion.MU_EARTH * 1e9, oe)
			location = np.array(r_orbit, dtype=np.float64)
			location /= np.linalg.norm(location)

			# Add up to 2 deg of off-track noise in a tangent direction.
			tangent = np.random.normal(size=3)
			tangent -= np.dot(tangent, location) * location
			tangent_norm = np.linalg.norm(tangent)
			if tangent_norm > 1e-12:
				tangent /= tangent_norm
				offset_mag = np.tan(np.radians(np.random.uniform(0.0, self.max_offset_deg)))
				location = location + offset_mag * tangent
				location /= np.linalg.norm(location)

			location *= self.radius
			self.targets.append(
				Target(
					name=f"tgt-{i}",
					r_LP_P=location,
					priority=self.priority_distribution(),
				)
			)


def build_targets(scene_module):
	"""Centralized target scenario selection for all envs.

	`paper` is the repo's paper-aligned mode.
	"""
	if target_distribution == "paper":
		return GroundTrackTargets(n_targets)
	if target_distribution == "uniform":
		return scene_module.UniformTargets(
			n_targets, priority_distribution=_paper_priority_distribution
		)
	if target_distribution == "cities":
		return scene_module.CityTargets(
			n_targets, priority_distribution=_paper_priority_distribution
		)
	raise ValueError(f"Unknown target_distribution: {target_distribution}")

SAT_ARGS = dict(
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    batteryStorageCapacity=120.0 * 3600,
    storedCharge_Init=120.0 * 3600,
    dataStorageCapacity=200 * 8e6 * 100,
    u_max=0.4,
    imageTargetMinimumElevation=np.arctan(800 / 500),
    K1=0.25,
    K3=3.0,
    omega_max=np.radians(5),
    servo_Ki=5.0,
    servo_P=150 / 5,
    oe=lambda: random_orbit(alt=orbit_alt_km),
)

SAT_ARGS_POWER = {}
SAT_ARGS_POWER.update(SAT_ARGS)
SAT_ARGS_POWER.update(
    dict(
        batteryStorageCapacity=180.0 * 3600,
        storedCharge_Init=lambda: 180.0 * 3600 * np.random.uniform(0.4, 0.8),
        rwBasePower=20.4,
        instrumentPowerDraw=-10.0,
        thrusterPowerDraw=-30.0,
        nHat_B=np.array([0, 0, -1]),
        maxWheelSpeed=6000.0,
        wheelSpeeds=lambda: np.random.uniform(-5000, 5000, 3),
        desatAttitude="nadir",
        storageInit=0,
        transmitterBaudRate=-50 * 8e6,
        transmitterPowerDraw=-25.0,
        instrumentBaudRate=5 * 8e6,
        basePowerDraw=-10.0,
        panelArea=0.55,
    )
)

from envs.charge import get_charge_env
from envs.desat import get_desat_env
from envs.downlink import get_downlink_env
from envs.resource import get_resource_env

__all__ = [
	"duration",
	"decision_interval",
	"orbit_alt_km",
	"target_distribution",
	"n_targets",
	"n_ahead",
	"build_targets",
    "SAT_ARGS",
    "SAT_ARGS_POWER",
	"get_charge_env",
	"get_desat_env",
	"get_downlink_env",
	"get_resource_env",
]
