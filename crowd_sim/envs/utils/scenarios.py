import logging
from enum import Enum
from abc import ABC

import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.human import Human


class Scenario(Enum):
    CIRCLE_CROSSING = "circle_crossing"
    CORRIDOR = "corridor"
    CORNER = "corner"
    T_INTERSECTION = "t_intersection"

    @classmethod
    def find(cls, name):
        for i in cls:
            if name in i.value:
                return i


class ScenarioConfig:
    def __init__(self, scenario, config):
        self.rng = np.random.default_rng()
        if type(scenario) == Scenario:
            self.scenario = scenario
        else:
            self.scenario = Scenario.find(scenario)
        self.configure(config)

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
        self.human_radius = config.humans.radius
        self.discomfort_dist = config.reward.discomfort_dist

    def get_spawn_position(self):  # return (center, goal), no noise
        if self.scenario == Scenario.CIRCLE_CROSSING:
            angle = self.rng.random() * np.pi * 2
            p = self.circle_radius * np.array([np.cos(angle), np.sin(angle)])
            return p, -p
        else:
            p_idx, g_idx = np.random.choice(range(len(self.spawn_positions)), 2, replace=False)
            return self.spawn_positions[p_idx], self.spawn_positions[g_idx]

    def get_spawn_positions(self, groups):
        num_human = sum(groups)
        average_human = num_human / len(self.spawn_positions)  # avg human per spawn pos
        # sample_radius = average_human * (self.human_radius * 2 + self.discomfort_dist)
        sample_radius = 1
        noise = self.rng.uniform(-sample_radius, sample_radius, 2)
        center, goal = self.get_spawn_position()
        return center + noise, goal + noise


class SceneManager(object):
    def __init__(self, scenario, robot, config):
        self.scenario_config = ScenarioConfig(scenario, config)
        self.configure(config)

        self.robot = robot
        self.humans = []
        self.membership = []

        self.rng = np.random.default_rng()

    def configure(self, config):
        self.config = config
        self.human_radius = self.config.humans.radius
        self.discomfort_dist = self.config.reward.discomfort_dist
        self.randomize_attributes = self.config.env.randomize_attributes

    def get_scene(self):
        group_membership = []
        individual_membership = []
        for group in self.membership:
            if len(group) == 1:
                individual_membership.append(*group)
            else:
                group_membership.append(group)
        return self.humans, self.scenario_config.obstacles, group_membership, individual_membership

    def spawn(self, num_human=5, group_size_lambda=1.2, use_groups=True):
        # group_sizes = self.rng.poisson(group_size_lambda, num_groups) + 1  # no zeros
        # human_idx = np.array(range(sum(group_sizes)))
        if use_groups:
            group_sizes = []
            while True:
                size = self.rng.poisson(group_size_lambda) + 1
                if sum(group_sizes) + size > num_human:
                    if num_human > sum(group_sizes):
                        group_sizes.append(num_human - sum(group_sizes))
                    break
                else:
                    group_sizes.append(size)
        else:
            group_sizes = np.ones(num_human)
        human_idx = np.arange(num_human)
        self.membership = self.split_array(human_idx, group_sizes)

        print(group_sizes)
        for i, size in enumerate(group_sizes):
            center, goal = self.scenario_config.get_spawn_positions(group_sizes)
            # print(size, center, goal)
            # print(self.humans)
            print(f"Spawn group {i} of size {size}, center: {center}, goal: {goal}")
            self.humans += self.spawn_group(size, center, goal)

    def spawn_group(self, size, center, goal):
        humans = []
        while True:
            noise = (
                (self.rng.random(2) - 0.5) * (self.human_radius * 2 + self.discomfort_dist) * size
            )
            spawn_pos = center + noise  # spawn noise based on group size
            if self.check_collision(spawn_pos, humans):  # break if there is collision
                continue

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
        assert type(array) == np.ndarray, "Input must be an ndarray"
        output_array = []
        idx = 0
        for s in split:
            output_array.append(array[idx : idx + s])
            idx += s
        return np.array(output_array)
