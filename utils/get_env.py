from envs import get_downlink_env, get_desat_env, get_charge_env, get_resource_env


def get_env(env_name):
    if env_name == "charge":
        env = get_charge_env()
    elif env_name == "resource":
        env = get_resource_env()
    elif env_name == "desat":
        env = get_desat_env()
    elif env_name == "downlink":
        env = get_downlink_env()
    else:
        raise NotImplementedError(f"{env_name} is not implemented.")
    return env
