from enum import Enum
from abc import ABC

import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.human import Human


class Scenario(Enum):
    CIRCLE_CROSSING = 1
    CORRIDOR = 2
    CORNER = 3
    T_INTERSECTION = 4


class ScenrioConfig:
    def __init__(self, scenario, config):
        self.rng = np.random.default_rng()
        self.scenario = scenario
        self.obstacles = []
        self.spawn_positions = []
        self.v_pref = config.humans.v_pref

        # if self.scenario == Scenario.CIRCLE_CROSSING:

    def get_spawn_position(self):
        if self.scenario == Scenario.CIRCLE_CROSSING:
            angle = self.rng.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise = (self.rng.random(2) - 0.5) * self.v_pref
            p = self.circle_radius * np.array([np.cos(angle), np.sin(angle)]) + noise
            return p
        else:
            return np.random.choice(self.spawn_positions)


class SceneManager(object):
    def __init__(self, scenario, robot):
        self.rng = np.random.default_rng()
        self.robot = robot
        self.humans = []

    def configure(self, config):
        self.config = config
        self.human_radius = self.config.humans.radius
        self.discomfort_dist = self.config.reward.discomfort_dist
        self.randomize_attributes = self.config.env.randomize_attributes

    def spawn(self, num_groups=5, group_size_lambda=1.5):
        group_sizes = self.rng.poisson(group_size_lambda, num_groups) + 1  # no zeros
        humans = np.array(range(sum(group_sizes)))
        group_membership = self.split_array(humans, group_sizes)

    def spawn_group(self, size, center, goal):
        while True:
            humans = []
            noise = (self.rng.random(2) - 0.5) * self.human_radius * 2 * size
            spawn_pos = center + noise  # spawn noise based on group size
            if self.check_collision(spawn_pos, humans):  # break if there is collision
                break

            human = Human(self.config, "humans")
            if self.randomize_attributes:
                human.sample_random_attributes()
            human.set(*spawn_pos, *(goal + noise), 0, 0, 0)  # TODO: set theta?
            humans.append(human)
            if len(humans) == size:
                return humans

    def check_collision(self, position, others=[]):
        collide = False
        for agent in [self.robot] + self.humans + others:
            min_dist = self.human_radius + agent.radius + self.discomfort_dist
            if (
                norm((position - agent.get_position())) < min_dist
                or norm((position - agent.get_goal_position())) < min_dist
            ):
                collide = True
                break
        return collide

    @staticmethod
    def split_array(array, split):
        # split array into subarrays of variable lengths
        assert len(array) == sum(split), "sum of splits must equal to array length!"
        assert type(array) == np.ndarray, "Input must be a ndarray"
        output_array = []
        idx = 0
        for s in split:
            output_array.append(array[idx : idx + s])
            idx += s
        return np.array(output_array)
