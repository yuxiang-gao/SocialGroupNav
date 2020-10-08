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

# from crowd_nav.utils.trainer import VNRLTrainer, MPRLTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
import configparser

# %%
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

env_config_file = Path(__file__).parent / "crowd_nav/configs/corridor_0.config"
print(env_config_file)
env_config = configparser.RawConfigParser()
env_config.read(env_config_file)
env = gym.make("CrowdSim-v0")
env.configure(env_config)

# %%
robot = Robot(env_config, "robot")
robot.set_policy(policy_factory["socialforce"]())
robot.policy.multiagent_training = True
env.set_robot(robot)
# env.set_scene("corner")

rewards = []
env.human_num = 10
env.train_val_sim = "t_intersection"
ob = env.reset("val")
#%%
env.render()
