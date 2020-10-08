from gym.envs.registration import register
from warnings import warn

try:
    register(
        id="CrowdSim-v0", entry_point="crowd_sim.envs:CrowdSim",
    )
except:
    warn("Environment is already registered.")
