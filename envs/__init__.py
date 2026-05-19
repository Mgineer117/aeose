duration = 5700.0 * 5  # 5 orbits
target_distribution = "uniform"
n_targets = 135
n_ahead = 3

from envs.charge import get_charge_env
from envs.desat import get_desat_env
from envs.downlink import get_downlink_env
from envs.resource import get_resource_env

__all__ = [
	"duration",
	"target_distribution",
	"n_targets",
	"n_ahead",
	"get_charge_env",
	"get_desat_env",
	"get_downlink_env",
	"get_resource_env",
]
