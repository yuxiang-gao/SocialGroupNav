import logging
from enum import Enum
from abc import ABC

import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.utils import point_to_segment_dist


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
    def __init__(self, scenario, config, seed):
        self.rng = np.random.default_rng(seed)
        if type(scenario) == Scenario:
            self.scenario = scenario
        else:
            self.scenario = Scenario.find(scenario)
        self.configure(config)

        self.obstacles = []
        self.spawn_positions = []
        self.width = 7

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
            self.spawn_positions = [[length, 0], [-length, 0]]
            self.robot_spawn_positions = [[length+1, -1], [-length-1, +1]]

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
            self.robot_spawn_positions = [
                [-length-1, self.width / 2+1],
                [length+1, self.width / 2+1],
                [1, -length-1],
            ]

    def configure(self, config):
        self.v_pref = config.getfloat("humans", "v_pref")
        self.circle_radius = config.getfloat("sim", "circle_radius")
        self.human_radius = config.getfloat("humans", "radius")
        self.discomfort_dist = config.getfloat("reward", "discomfort_dist")

    def get_robot_spawn_position(self):
        p_idx, g_idx = np.random.choice(range(len(self.robot_spawn_positions)), 2, replace=False)
        return self.robot_spawn_positions[p_idx], self.robot_spawn_positions[g_idx]

    def get_spawn_position(self):  # return (center, goal), no noise
        if self.scenario == Scenario.CIRCLE_CROSSING:
            angle = self.rng.random() * np.pi * 2
            p = self.circle_radius * np.array([np.cos(angle), np.sin(angle)])
            return p, -p
        else:
            p_idx, g_idx = np.random.choice(range(len(self.spawn_positions)), 2, replace=False)
            return self.spawn_positions[p_idx], self.spawn_positions[g_idx]

    def get_spawn_positions(self, groups=None):
        if self.scenario == Scenario.CIRCLE_CROSSING:
            return self.get_spawn_position()
        else:
            # num_human = sum(groups)
            # average_human = num_human / len(self.spawn_positions)  # avg human per spawn pos
            # sample_radius = average_human * (self.human_radius * 2 + self.discomfort_dist)
            sample_radius = 1
            noise = self.rng.uniform(-sample_radius, sample_radius, 2)

        center, goal = self.get_spawn_position()
        return center + noise, goal + noise


class SceneManager(object):
    def __init__(self, scenario, robot, config, seed):
        self.humans = []
        self.membership = []
        self.scenario_config = None
        # self.obstacles_sampled = None
        self.rng = None


        self.configure(config)
        self.set_scenario(scenario, seed)
        self.robot = robot


    def configure(self, config):
        self.config = config
        self.human_radius = self.config.getfloat("humans", "radius")
        self.robot_radius = self.config.getfloat("robot", "radius")
        self.discomfort_dist = self.config.getfloat("reward", "discomfort_dist")
        self.randomize_attributes = self.config.getboolean("env", "randomize_attributes")

    def set_scenario(self, scenario, seed):
        self.humans = []
        self.membership = []
        self.scenario_config = ScenarioConfig(scenario, self.config, seed)
        # pass obstacles to Agent class
        JointState.obstacles = self.get_obstacles()
        self.rng = np.random.default_rng(seed=seed)
        # self.obstacles_sampled = self.sample_obstalces(self.get_obstacles())

    def get_current_scenario(self):
        return self.scenario_config.scenario.value

    def get_scene(self):
        group_membership = []
        individual_membership = []
        for group in self.membership:
            if len(group) == 1:
                individual_membership.append(*group)
            else:
                group_membership.append(group)
        return self.humans, self.scenario_config.obstacles, group_membership, individual_membership

    def get_obstacles(self):
        return self.scenario_config.obstacles

    def spawn(self, num_human=5, group_size_lambda=1.2, use_groups=True, set_robot=True):
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

        # Spawn robot
        if set_robot:
            center, goal = self.scenario_config.get_robot_spawn_position()
            logging.info(f"Spawn robot, center: {center}, goal: {goal}")
            self.spawn_robot(center, goal)

        logging.info(f"Generating groups of size: {group_sizes}")
        for i, size in enumerate(group_sizes):
            center, goal = self.scenario_config.get_spawn_positions(group_sizes)
            logging.info(f"Spawn group {i} of size {size}, center: {center}, goal: {goal}")
            self.humans += self.spawn_group(size, center, goal)

    def spawn_robot(self, center, goal):
        noise = (self.rng.random(2) - 0.5) * (self.robot_radius * 2 + self.discomfort_dist)
        spawn_pos = center + noise  # spawn noise based on group size
        agent_radius = self.robot_radius
        while True:
            if not self.check_collision(spawn_pos, robot=True):  # break if there is no collision
                break
            else:
                spawn_pos += (
                    self.rng.random(2) - 0.5
                ) * agent_radius  # gentlely nudge the new ped to avoid collision
        self.robot.set(*spawn_pos, *goal, 0, 0, 0)

    def spawn_group(self, size, center, goal):
        humans = []
        while True:
            noise = (
                (self.rng.random(2) - 0.5) * (self.human_radius * 2 + self.discomfort_dist) * size
            )
            spawn_pos = center + noise  # spawn noise based on group size
            while True:
                if not self.check_collision(spawn_pos, humans):  # break if there is no collision
                    break
                else:
                    spawn_pos += (
                        self.rng.random(2) - 0.5
                    ) * self.human_radius  # gentlely nudge the new ped to avoid collision

            human = Human(self.config, "humans")
            if self.randomize_attributes:
                human.sample_random_attributes()
            human.set(*spawn_pos, *(goal), 0, 0, 0)  # TODO: set theta?
            humans.append(human)
            if len(humans) == size:
                return humans

    def check_collision(self, position, others=[], robot=False):
        """Check collision, return true if there is collsion, false otherwise"""

        agents = [self.robot] + self.humans + others if not robot else self.humans + others
        radius = self.human_radius if not robot else self.robot_radius
        # Check collision with agents
        for agent in agents:
            min_dist = radius + agent.radius + self.discomfort_dist
            if norm((position - agent.get_position())) < min_dist:
                return True

        # Check with obstacles
        for ob in self.get_obstacles():
            min_dist = self.human_radius + self.discomfort_dist
            if self.line_distance(ob, position) < min_dist:
                return True
        return False

    @staticmethod
    def line_distance(line, point):
        """Euclidean distance between a point and a line (x_min, x_max, y_min,  y_max)"""
        endpoints = np.array(line)[[0, 2, 1, 3]]
        return point_to_segment_dist(*endpoints, *point)

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

    @staticmethod
    def sample_obstalces(ob, resolution=10):
        """Sample from line obstacles"""
        ob_sampled = None
        for startx, endx, starty, endy in ob:
            samples = int(norm((startx - endx, starty - endy)) * resolution)
            line = np.array(
                list(zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples)))
            )
            ob_sampled = line if ob_sampled is None else np.vstack((ob_sampled, line))
        return ob_sampled
