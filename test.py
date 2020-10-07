# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import copy
import git
import re
from tensorboardX import SummaryWriter
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import VNRLTrainer, MPRLTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


# %%
from crowd_nav.configs.icra_benchmark.config import (
    BaseEnvConfig,
    BasePolicyConfig,
    BaseTrainConfig,
    Config,
)


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.env.randomize_attributes = True
        self.env.time_step = 0.25
        self.sim.centralized_planning = False
        self.sim.test_scenario = "t_intersection"
        self.sim.human_num = 10
        self.humans.policy = "socialforce"


env_config = EnvConfig(True)
env = gym.make("CrowdSim-v0")
env.configure(env_config)
robot = Robot(env_config, "robot")
robot.kinematics = "holonomic"
robot.time_step = env.time_step
robot.policy = policy_factory["socialforce"]()
robot.policy.multiagent_training = True
env.set_robot(robot)
# env.set_scene("corner")


# %%
rewards = []
ob = env.reset()
env.render("traj")
