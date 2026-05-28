from envs import get_downlink_env, get_desat_env, get_charge_env, get_resource_env
from utils.wrapper import ActionSuccessWrapper


def get_env(env_name, n_ahead=None):
    if env_name == "charge":
        env = get_charge_env(n_ahead=n_ahead) if n_ahead is not None else get_charge_env()
    elif env_name == "resource":
        env = (
            get_resource_env(n_ahead=n_ahead)
            if n_ahead is not None
            else get_resource_env()
        )
    elif env_name == "desat":
        env = get_desat_env(n_ahead=n_ahead) if n_ahead is not None else get_desat_env()
    elif env_name == "downlink":
        env = (
            get_downlink_env(n_ahead=n_ahead)
            if n_ahead is not None
            else get_downlink_env()
        )
    else:
        raise NotImplementedError(f"{env_name} is not implemented.")
    
    env = ActionSuccessWrapper(env)
    return env
