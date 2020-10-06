import logging
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


class ScenarioConfig:
    def __init__(self, scenario):
        self.rng = np.random.default_rng()
        self.scenario = scenario
        self.obstacles = []
        self.spawn_positions = []
        self.width = 5

        if self.scenario == Scenario.CORNER:
            self.obstacles = np.array(
                [
                    [-self.width, self.width, self.width, self.width],
                    [-self.width, 0, 0, 0],
                    [self.width, self.width, -self.width, self.width],
                    [0, 0, -self.width, 0],
                ]
            )
            self.spawn_positions = [[-self.width, self.width / 2], [self.width / 2, -self.width]]
        elif self.scenario == Scenario.CORRIDOR:
            length = 5
            self.obstacles = np.array(
                [
                    [-length, length, self.width / 2, self.width / 2],
                    [-length, length, -self.width / 2, -self.width / 2],
                ]
            )
            self.spawn_positions = [[length, 1], [length, -1], [-length, 1], [-length, -1]]

        elif self.scenario == Scenario.T_INTERSECTION:
            length = 5
            self.obstacles = [
                [-length, length, self.width, self.width],
                [-length, -self.width / 2, 0, 0],
                [self.width / 2, length, 0, 0],
                [self.width / 2, self.width / 2, 0, -length],
                [-self.width / 2, -self.width / 2, 0, -length],
            ]
            self.spawn_positions = [
                [-length, self.width / 2],
                [length, self.width / 2],
                [0, -length],
            ]

    def configure(self, config):
        self.v_pref = config.humans.v_pref
        self.circle_radius = config.sim.circle_radius
        self.human_radius = self.config.humans.radius
        self.discomfort_dist = self.config.reward.discomfort_dist

    def get_spawn_position(self):  # return (center, goal), no noise
        if self.scenario == Scenario.CIRCLE_CROSSING:
            angle = self.rng.random() * np.pi * 2
            p = self.circle_radius * np.array([np.cos(angle), np.sin(angle)])
            return p, -p
        else:
            return np.random.choice(self.spawn_positions, 2, replace=False)

    def get_spawn_positions(self, groups):
        num_human = sum(groups)
        average_human = num_human / len(self.spawn_positions)  # avg human per spawn pos
        sample_radius = average_human * (self.human_radius * 2 + self.discomfort_dist)
        noise = self.rng.uniform(-sample_radius, sample_radius, 2)
        center, goal = self.get_spawn_position()
        return center + noise, goal + noise


class SceneManager(object):
    def __init__(self, scenario, robot):
        self.scenario_config = ScenarioConfig(scenario)
        self.robot = robot
        self.humans = []
        self.group_membership = []

        self.rng = np.random.default_rng()

    def configure(self, config):
        self.scenario_config.configure(config)
        self.config = config
        self.human_radius = self.config.humans.radius
        self.discomfort_dist = self.config.reward.discomfort_dist
        self.randomize_attributes = self.config.env.randomize_attributes

    def spawn(self, num_groups=5, group_size_lambda=1.5):
        group_sizes = self.rng.poisson(group_size_lambda, num_groups) + 1  # no zeros
        human_idx = np.array(range(sum(group_sizes)))
        self.group_membership = self.split_array(human_idx, group_sizes)

        for i, size in enumerate(group_sizes):
            center, goal = self.scenario_config.get_spawn_positions(group_sizes)
            self.humans += self.spawn_group(size, center, goal)
            logging.info(f"Spawn group {i}, center: {center}, goal: {goal}")

    def spawn_group(self, size, center, goal):
        humans = []
        while True:
            noise = (
                (self.rng.random(2) - 0.5) * (self.human_radius * 2 + self.discomfort_dist) * size
            )
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

    def check_collision(self, position, others=[]):  # TODO: check obstacles
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
