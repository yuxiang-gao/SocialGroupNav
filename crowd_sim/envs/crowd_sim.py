import logging
import gym
import rvo2
import numpy as np
import matplotlib.lines as mlines
from warnings import warn
from numpy.linalg import norm
from matplotlib import patches
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from math import sqrt
from crowd_sim.envs.policy.policy_factory import policy_factory

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # Simulation configuration
        self.config = None
        self.time_limit = None
        self.time_step = None
        self.end_on_collision = True
        self.side = None
        self.pixel_side = None
        self.closed = None
        self.goal_radius = None
        self.max_humans = None
        self.min_humans = None
        self.human_num_mode = None
        self.human_num = None
        self.perpetual = None
        self.rotate_path = None
        self.randomize_attributes = None
        self.square_width = None
        self.circle_radius = None
        # Reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_scale = None
        self.discomfort_penalty_factor = None
        self.group_discomfort_penalty = None
        self.time_penalty = None
        self.progress_reward = None
        self.initial_distance = None
        self.previous_distance = None
        # Internal environment configuration
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.parallel = None
        self.max_tries = None
        self.train_val_sim = None
        self.test_sim = None
        # For visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        # For information return
        self.obs_history = np.array([])
        self.episode_info = dict()
        self.movie_file = ''

        self.use_groups = None
        self.min_group_num = None
        self.max_group_num = None
        self.centralized_planning = None
        self.centralized_planner = None


        self.enable_intent = None
        self.intent_type = None

        self.obstacles = []  # xmin,xmax,ymin,ymax

    def configure(self, config):
        self.config = config
        # Simulation:
        self.time_limit = config.getint('env', 'time_limit', fallback=25)
        self.time_step = config.getfloat('env', 'time_step', fallback=0.25)
        self.end_on_collision = config.getboolean('env', 'end_on_collision')
        self.side = config.getfloat('env', 'side', fallback=12)
        self.pixel_side = config.getint('env', 'pixel_side', fallback=84)
        self.num_frames = config.getint('env', 'num_frames', fallback=4)
        self.obs_history = np.zeros((1, self.num_frames, self.pixel_side, self.pixel_side))
        self.closed = config.getboolean('env', 'closed', fallback=True)
        self.goal_radius = config.getfloat('env', 'goal_radius', fallback=0.3)
        self.max_humans = config.getfloat('env', 'max_humans')
        self.min_humans = config.getfloat('env', 'min_humans')

        self.use_groups = config.getboolean('sim', 'use_groups', fallback=False)
        self.min_group_num = config.getint('sim', 'min_group_num', fallback=4)
        self.max_group_num = config.getint('sim', 'max_group_num', fallback=4)

        self.num_groups = None

        self.centralized_planning = config.getboolean('sim', 'centralized_planning', fallback=False)

        human_policy = config.get('humans', 'policy')
        if self.centralized_planning:
            self.centralized_planner = policy_factory['centralized_' + human_policy]()



        self.human_num_mode = config.get('env', 'human_num_mode', fallback='gauss')
        self.perpetual = config.getboolean('env', 'perpetual', fallback=True)
        self.rotate_path = config.getboolean('env', 'rotate_path', fallback=True)
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        # Reward:
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_scale = config.getfloat('reward', 'discomfort_scale', fallback=1.0)
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.group_discomfort_penalty = config.getfloat('reward', 'group_discomfort_penalty')
        self.time_penalty = config.getfloat('reward', 'time_penalty')
        self.progress_reward = config.getfloat('reward', 'progress_reward')
        # Internal environment configuration
        self.parallel = config.getboolean('env', 'parallel', fallback=True)
        self.max_tries = 100

        self.enable_intent = config.getboolean('env', 'enable_intent', fallback=False)
        self.intent_type = config.get('env', 'intent_type', fallback='individual') # or 'group'

        if self.config.get('humans', 'policy') == 'orca' or self.config.get('humans', 'policy') == 'socialforce' or self.config.get('humans', 'policy') == 'centralized_socialforce':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def set_obstacles(self, obs):
        self.obstacles = obs

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == "corridor":
            self.set_obstacles([[-5, 5, 2, 2], [-5, 5, -2, -2]])
            sign = np.random.choice([-1, 1])

            while True:
                self.humans = []
                angle = np.random.random() * np.pi * 2

                for i in range(human_num):
                    human = Human(self.config, "humans")
                    if self.randomize_attributes:
                        human.sample_random_attributes()

                    # add some noise to simulate all the possible cases robot could meet with human
                    px_noise = (np.random.random() - 0.5) * human.v_pref
                    py_noise = (np.random.random() - 0.5) * human.v_pref
                    px = 5 * sign + px_noise
                    py = 0 + py_noise
                    human.set(px, py, -px, py, 0, 0, 0)
                    self.humans.append(human)

                # check for collisions
                for human in self.humans:
                    collide = False
                    for agent in [self.robot] + self.humans:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if (
                                norm((px - agent.px, py - agent.py)) < min_dist
                                or norm((px - agent.gx, py - agent.gy)) < min_dist
                        ):
                            collide = True
                            break

                if not collide:
                    break
                else:
                    print ("Collision")
            print ("Done adding humans")

        elif rule == "corner":
            width = 5
            self.set_obstacles(
                [
                    [-width, width, width, width],
                    [-width, 0, 0, 0],
                    [width, width, -width, width],
                    [0, 0, -width, 0],
                ]
            )
            center = np.random.choice([[-width, width / 2], [width / 2, -width]])  # choose a center
            while True:
                self.humans = []
                angle = np.random.random() * np.pi * 2

                for i in range(human_num):
                    human = Human(self.config, "humans")
                    if self.randomize_attributes:
                        human.sample_random_attributes()
                    px_noise = (np.random.random() - 0.5) * human.v_pref
                    py_noise = (np.random.random() - 0.5) * human.v_pref
                    px = center[0] + px_noise
                    py = center[1] + py_noise
                    human.set(px, py, -px, py, 0, 0, 0)
                    self.humans.append(human)

                # check for collisions
                for human in self.humans:
                    collide = False
                    for agent in [self.robot] + self.humans:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if (
                                norm((px - agent.px, py - agent.py)) < min_dist
                                or norm((px - agent.gx, py - agent.gy)) < min_dist
                        ):
                            collide = True
                            break

                if not collide:
                    break
        elif rule == "t_intersection":
            width = 5
            self.set_obstacles(
                [
                    [-width, width, width, width],
                    [-width, -width / 2, 0, 0],
                    [width / 2, width, 0, 0],
                    [width / 2, width / 2, 0, -width],
                    [-width / 2, -width / 2, 0, -width],
                ]
            )
            center = np.random.choice(
                [[-width, width / 2], [width, width / 2], [0, -width]]
            )  # choose a center
            while True:
                self.humans = []
                angle = np.random.random() * np.pi * 2

                for i in range(human_num):
                    human = Human(self.config, "humans")
                    if self.randomize_attributes:
                        human.sample_random_attributes()
                    px_noise = (np.random.random() - 0.5) * human.v_pref
                    py_noise = (np.random.random() - 0.5) * human.v_pref
                    px = center[0] + px_noise
                    py = center[1] + py_noise
                    human.set(px, py, -px, py, 0, 0, 0)
                    self.humans.append(human)

                # check for collisions
                for human in self.humans:
                    collide = False
                    for agent in [self.robot] + self.humans:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if (
                                norm((px - agent.px, py - agent.py)) < min_dist
                                or norm((px - agent.gx, py - agent.gy)) < min_dist
                        ):
                            collide = True
                            break

                if not collide:
                    break


        elif rule == 'mixed':
            # mix different training simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    tries = 0
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        if norm((px - self.robot.gx,
                                 py - self.robot.gy)) < human.radius + self.goal_radius + self.discomfort_dist:
                            collide = True
                        else:
                            for agent in [self.robot] + self.humans:
                                if norm((px - agent.px, py - agent.py)) < \
                                        human.radius + agent.radius + self.discomfort_dist:
                                    collide = True
                                    break
                        if not collide or tries > self.max_tries:
                            if tries > self.max_tries:
                                warn('Exceeding max_tries in generate_random_human_position (static case)')
                            break
                        sign *= -1
                        tries += 1
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 humans will be in the circle crossing scenarios
                # the rest of the humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        elif rule == "really_mixed":
            self.humans = []
            for i in range(human_num):
                discriminator = np.random.rand()
                if discriminator < 0.4:
                    self.humans.append(self.generate_square_crossing_human())
                elif 0.4 < discriminator <= 0.8:
                    self.humans.append(self.generate_circle_crossing_human())
                else:
                    self.humans.append(self.generate_static_obstacle())
        else:
            raise ValueError("Rule doesn't exist")

    def generate_static_obstacle(self):
        """  Generates a static obstacle  """
        width = 4
        height = 8
        human = Human(self.config, 'humans')
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        tries = 0
        while True:
            px = np.random.random() * width * 0.5 * sign
            py = (np.random.random() - 0.5) * height
            collide = False
            if norm((px - self.robot.gx, py - self.robot.gy)) < human.radius + self.goal_radius + self.discomfort_dist:
                collide = True
            else:
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
            if not collide or tries > self.max_tries:
                if tries > self.max_tries:
                    warn('Exceeding max_tries in generate_static_obstacle')
                break
            else:
                sign *= -1
            tries += 1
        human.set(px, py, px, py, 0, 0, 0)
        return human

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        tries = 0
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            if norm((px - self.robot.gx, py - self.robot.gy)) < human.radius + self.goal_radius + self.discomfort_dist:
                collide = True
            else:
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
            if not collide or tries > self.max_tries:
                if tries > self.max_tries:
                    warn('Exceeding max_tries in generate_circle_crossing_human')
                break
            else:
                if self.randomize_attributes:
                    human.sample_random_attributes()
            tries += 1
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        tries = 0
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            if norm((px - self.robot.gx, py - self.robot.gy)) < human.radius + self.goal_radius + self.discomfort_dist:
                collide = True
            else:
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
            if not collide or tries > self.max_tries:
                if tries > self.max_tries:
                    warn('Exceeding max_tries in generate_square_crossing_human, conditional 1')
                break
            sign *= -1
            tries += 1
        tries = 0
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide or tries > self.max_tries:
                if tries > self.max_tries:
                    warn('Exceeding max_tries in generate_square_crossing_human, conditional 2')
                break
            sign *= -1
            tries += 1
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_humans_in_groups(self, human_num, rule):
        self.humans = []
        self.group_membership = []
        self.individual_membership = []

        # randomly select number of groups
        self.num_groups = np.random.randint(low=self.min_group_num, high=self.max_group_num+1)

        # randomly assign number of humans to groups, last value means no group association
        group_dict = {human_list: 0 for human_list in range(self.num_groups+1)}

        for i in range(human_num):
            group_id = np.random.randint(low=0, high=self.num_groups+1)
            group_dict[group_id] += 1

        # group members should have colocated starting and ending positions
        idx = 0
        for i in range(self.num_groups):
            if group_dict[i] > 0:
                humans_in_group = self.generate_n_humans_in_group(group_dict[i], rule)
                self.humans = self.humans + humans_in_group
                group_members = []
                for _ in range(group_dict[i]):
                    group_members.append(idx)
                    idx += 1

                self.group_membership.append(group_members)

        # for each human in last group, treat as individual
        for i in range(group_dict[self.num_groups]):
            humans_in_group = self.generate_n_humans_in_group(1, rule)
            self.humans = self.humans + humans_in_group
            self.individual_membership.append(idx)
            idx += 1

    # generate n humans in a group that have co-located starting and end positions
    def generate_n_humans_in_group(self, humans_in_group, rule):
        if rule == 'circle_crossing':

            while True:
                humans = []
                angle = np.random.random() * np.pi * 2

                for i in range(humans_in_group):
                    human = Human(self.config, 'humans')
                    if self.randomize_attributes:
                        human.sample_random_attributes()

                    # add some noise to simulate all the possible cases robot could meet with human
                    px_noise = (np.random.random() - 0.5) * human.v_pref
                    py_noise = (np.random.random() - 0.5) * human.v_pref
                    px = self.circle_radius * np.cos(angle) + px_noise
                    py = self.circle_radius * np.sin(angle) + py_noise
                    human.set(px, py, -px, -py, 0, 0, 0)
                    humans.append(human)


                # check for collisions
                for human in humans:
                    collide = False
                    for agent in [self.robot] + self.humans:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if (
                            norm((px - agent.px, py - agent.py)) < min_dist
                            or norm((px - agent.gx, py - agent.gy)) < min_dist
                        ):
                            collide = True
                            break

                if not collide:
                    break

        elif rule == "corridor":
            self.set_obstacles([[-5, 5, 2, 2], [-5, 5, -2, -2]])
            while True:
                if np.random.random() > 0.5:
                    sign = -1
                else:
                    sign = 1
                humans = []
                angle = np.random.random() * np.pi * 2

                for i in range(humans_in_group):
                    human = Human(self.config, "humans")
                    if self.randomize_attributes:
                        human.sample_random_attributes()

                    # add some noise to simulate all the possible cases robot could meet with human
                    px_noise = (np.random.random() - 0.5) * human.v_pref
                    py_noise = (np.random.random() - 0.5) * human.v_pref
                    # px = 5 + px_noise
                    # py = 0 + py_noise
                    px = self.circle_radius * np.cos(angle) + px_noise
                    py = self.circle_radius * np.sin(angle) + py_noise
                    human.set(px, py, -px, py, 0, 0, 0)
                    humans.append(human)


                # check for collisions
                for human in humans:
                    collide = False
                    for agent in [self.robot] + self.humans:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((px - agent.px, py - agent.py)) < min_dist or \
                                norm((px - agent.gx, py - agent.gy)) < min_dist:
                            collide = True
                            break

                if not collide:
                    break

        return humans


    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).
        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())
        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and \
                        (human.reached_destination() or
                         self.global_time >= self.time_limit - self.time_step):
                    self.human_times[i] = self.global_time
            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        del sim
        return self.human_times

    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if self.max_humans > 0:
            if self.human_num_mode == 'uniform':
                self.human_num = np.random.randint(self.min_humans, self.max_humans + 1)  # Vary the number of humans
            elif self.human_num_mode == 'gauss':
                self.human_num = int(np.round(np.random.randn()*(self.max_humans - self.min_humans)/2 +
                                              (self.max_humans + self.min_humans)/2))
                self.human_num = max(self.human_num, 3)
                self.human_num = min(self.human_num, 22)
            else:
                raise ValueError('Unknown human_num_mode')
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'
        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            agent_angle = np.pi*2*np.random.rand()
            if self.robot.sensor == 'coordinates' or self.rotate_path == False:
                self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi/2)
            else:
                px_i = self.circle_radius*np.cos(agent_angle)
                py_i = self.circle_radius*np.sin(agent_angle)
                gx_i = self.circle_radius*np.cos(agent_angle+np.pi)
                gy_i = self.circle_radius*np.sin(agent_angle+np.pi)
                self.robot.set(px_i, py_i, gx_i, gy_i, 0, 0, 0)
            if self.case_counter[phase] >= 0:
                if not self.parallel:
                    np.random.seed(counter_offset[phase] + self.case_counter[phase])

                if self.use_groups:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_humans_in_groups(human_num, rule=self.train_val_sim)

                else:
                    if phase in ['train', 'val']:
                        human_num = self.human_num if self.robot.policy.multiagent_training else 1
                        self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                    else:
                        self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = [np.zeros(self.human_num)]
        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor.lower() == 'rgb' or self.robot.sensor.lower() == 'gray':
            snapshot = self.get_pixel_obs()
            self.obs_history = snapshot
            for _ in range(self.num_frames - 1):
                self.obs_history = np.concatenate((self.obs_history, snapshot), axis=1)
            ob = self.obs_history
        else:
            raise ValueError('Unknown robot sensor type.')
        self.initial_distance = np.linalg.norm([(self.robot.px - self.robot.get_goal_position()[0]),
                                                (self.robot.py - self.robot.get_goal_position()[1])])
        self.previous_distance = self.initial_distance
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        # info contains the various contributions to the reward:
        self.episode_info = {'collisions': 0., 'time': 0., 'discomfort': 0., 'progress': 0.,
                             'goal': 0., 'group_discomfort': 0., 'global_time': 0., 'did_timeout': 0.,
                             'did_collide': 0., 'did_succeed': 0.}
        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def point_in_hull(self, point, hull, tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)

    def point_along_circle(self, h, k, r):
        theta = np.random.random() * 2 * np.pi
        return h + np.cos(theta) * r, k + np.sin(theta) * r


    # helper functions for minimum spanning circle

    # Python3 program to find the minimum enclosing
    # circle for N integer points in a 2-D plane
    from math import sqrt

    # Defining infinity
    INF = 10 ** 18

    # Function to return the euclidean distance
    # between two points
    def dist(a, b):
        return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

        # Function to check whether a point lies inside

    # or on the boundaries of the circle
    def is_inside(c, p):
        return CrowdSim.dist(c[0], p) <= c[1]

        # The following two functions are the functions used

    # To find the equation of the circle when three
    # points are given.

    # Helper method to get a circle defined by 3 points
    @staticmethod
    def get_circle_center(bx, by, cx, cy):
        B = bx * bx + by * by
        C = cx * cx + cy * cy
        D = bx * cy - by * cx
        return [(cy * B - by * C) // (2 * D),
                (bx * C - cx * B) // (2 * D)]

        # Function to return a unique circle that intersects

    # three points
    @staticmethod
    def circle_frOm(A, B, C):
        I = CrowdSim.get_circle_center(B[0] - A[0], B[1] - A[1],
                              C[0] - A[0], C[1] - A[1])
        I[0] += A[0]
        I[1] += A[1]
        return [I, CrowdSim.dist(I, A)]

        # Function to return the smallest circle

    # that intersects 2 points
    @staticmethod
    def circle_from(A, B):

        # Set the center to be the midpoint of A and B
        C = [(A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0]

        # Set the radius to be half the distance AB
        return [C, CrowdSim.dist(A, B) / 2.0]

        # Function to check whether a circle encloses the given points

    @staticmethod
    def is_valid_circle(c, P):

        # Iterating through all the points to check
        # whether the points lie inside the circle or not
        for p in P:
            if (CrowdSim.is_inside(c, p) == False):
                return False
        return True

    # Function to return find the minimum enclosing
    # circle from the given set of points
    @staticmethod
    def minimum_enclosing_circle(P):

        # To find the number of points
        n = len(P)

        if (n == 0):
            return [[0, 0], 0]
        if (n == 1):
            return [P[0], 0]

            # Set initial MEC to have infinity radius
        mec = [[0, 0], float('inf')]

        # Go over all pair of points
        for i in range(n):
            for j in range(i + 1, n):

                # Get the smallest circle that
                # intersects P[i] and P[j]
                tmp = CrowdSim.circle_from(P[i], P[j])

                # Update MEC if tmp encloses all points
                # and has a smaller radius
                if (tmp[1] < mec[1] and CrowdSim.is_valid_circle(tmp, P)):
                    mec = tmp

                    # Go over all triples of points
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):

                    # Get the circle that intersects P[i], P[j], P[k]
                    tmp = CrowdSim.circle_frOm(P[i], P[j], P[k])

                    # Update MEC if tmp encloses all points
                    # and has smaller radius
                    if (tmp[1] < mec[1] and CrowdSim.is_valid_circle(tmp, P)):
                        mec = tmp

        return mec

    @staticmethod
    def is_collision_with_circle(circle_x, circle_y, rad, x, y):

        # Compare radius of circle
        # with distance of its center
        # from given point
        if ((x - circle_x) * (x - circle_x) +
                (y - circle_y) * (y - circle_y) <= rad * rad):
            return True;
        else:
            return False;

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states, self.group_membership)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states, self.group_membership)
        else:
            human_actions = []
            for human in self.humans:
                # Choose new target if human has reached goal and in perpetual mode:
                if human.reached_destination() and self.perpetual:
                    if self.train_val_sim == 'square_crossing':
                        gx = np.random.random() * self.square_width * 0.5 * np.random.choice([-1, 1])
                        gy = (np.random.random() - 0.5) * self.square_width
                        human.set(human.px, human.py, gx, gy, 0, 0, 0)
                    elif self.train_val_sim == 'circle_crossing':
                        human.set(human.px, human.py, -human.px, -human.py, 0, 0, 0)
                    else:
                        if np.random.rand(1) > 0.5:
                            gx = np.random.random() * self.square_width * 0.5 * np.random.choice([-1, 1])
                            gy = (np.random.random() - 0.5) * self.square_width
                            human.set(human.px, human.py, gx, gy, 0, 0, 0)
                        else:
                            human.set(human.px, human.py, -human.px, -human.py, 0, 0, 0)
                # observation for humans is always coordinates
                human_ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
                if self.robot.visible:
                    human_ob += [self.robot.get_observable_state()]
                human_actions.append(human.act(human_ob, self.group_membership))
        # collision detection
        dmin = float('inf')
        collisions = 0
        human_distances = list()
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            human_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if human_dist < 0:
                collisions += 1
                self.episode_info['collisions'] -= self.collision_penalty
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, human_dist))
                break
            elif human_dist < dmin:
                dmin = human_dist
            human_distances.append(human_dist)
        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')
        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step, self.closed))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < \
            self.robot.radius + self.goal_radius
        done = False
        info = Nothing()
        reward = -self.time_penalty
        goal_distance = np.linalg.norm([(end_position[0] - self.robot.get_goal_position()[0]),
                                        (end_position[1] - self.robot.get_goal_position()[1])])
        progress = self.previous_distance - goal_distance
        self.previous_distance = goal_distance
        reward += self.progress_reward * progress
        self.episode_info['progress'] += self.progress_reward * progress
        if self.global_time >= self.time_limit:
            done = True
            info = Timeout()
            self.episode_info['did_succeed'] = 0.0
            self.episode_info['did_collide'] = 0.0
            self.episode_info['did_timeout'] = 1.0
        if collisions > 0:
            reward -= self.collision_penalty*collisions
            if self.end_on_collision:
                done = True
            info = Collision()
            self.episode_info['did_succeed'] = 0.0
            self.episode_info['did_collide'] = 1.0
            self.episode_info['did_timeout'] = 0.0
        if reaching_goal:
            reward += self.success_reward
            done = True
            info = ReachGoal()
            self.episode_info['goal'] = self.success_reward
            self.episode_info['did_succeed'] = 1.0
            self.episode_info['did_collide'] = 0.0
            self.episode_info['did_timeout'] = 0.0
        for human_dist in human_distances:
            if 0 <= human_dist < self.discomfort_dist * self.discomfort_scale:
                discomfort = (human_dist - self.discomfort_dist*self.discomfort_scale)\
                    * self.discomfort_penalty_factor * self.time_step
                reward += discomfort
                self.episode_info['discomfort'] += discomfort


        forces = self.centralized_planner.get_forces()

        # get group cohesive force
        total_group_cohesive_force = 0
        total_group_gaze_force = 0

        if forces is not None:
            grp_cohesive_force = forces[3].get_force() #TODO remove this hard coded index
            total_group_cohesive_force = np.sum(np.abs(grp_cohesive_force)) # TODO: verify from Yuxiang if we should normalize by numgroups
            total_group_cohesive_force = total_group_cohesive_force / self.num_groups

            grp_gaze_force = forces[5].get_force()
            total_group_gaze_force = np.sum(np.abs(grp_gaze_force))
            total_group_gaze_force = total_group_gaze_force / self.num_groups

        self.episode_info['group_cohesive_force'] = total_group_cohesive_force
        self.episode_info['group_gaze_force'] = total_group_gaze_force

        # penalize group intersection
        robot_pos = [self.robot.px, self.robot.py]

        convex = 1

        for group in self.group_membership:
            # get the members of the group
            points = []
            for human_id in group:
                ind_points = [self.point_along_circle(self.humans[human_id].px,self.humans[human_id].py,
                                                      self.humans[human_id].radius) for _ in range(10)]
                points.extend(ind_points)

            if convex == 1:

                # compute the convex hull
                hull = ConvexHull(points)

                group_col = self.point_in_hull(robot_pos, hull)

            # min spanning circle
            else:
                circle_def = CrowdSim.minimum_enclosing_circle(points)

                group_col = CrowdSim.is_collision_with_circle(circle_def[0][0], circle_def[0][1],
                                                              circle_def[1], robot_pos[0], robot_pos[1])

            if group_col:
                group_discomfort = -self.group_discomfort_penalty
                reward += group_discomfort
                self.episode_info['group_discomfort'] += group_discomfort

        if len(human_distances) > 0 and 0 <= min(human_distances) < self.discomfort_dist * self.discomfort_scale:
            info = Danger(min(human_distances))
        if update:
            # update all agents
            self.robot.step(action, self.closed)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action, self.closed)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time
            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]

                if self.enable_intent:
                    if self.intent_type == 'individual':
                        target_maps = np.array([human.get_target_map() for human in self.humans])
                    elif self.intent_type == 'group':
                        target_maps = np.array([human.get_target_map() for human in self.humans])

                        # average intent map across group members
                        for group in self.group_membership:
                            # get the members of the group
                            avg = np.average([target_maps[human_id] for human_id in group], axis=0)
                            for human_id in group:
                                target_maps[human_id] = avg

                        # add target_map to observation
                        for i in range(len(ob)):
                            ob[i].update_target_map(target_maps[i])
                    else:
                        print("unrecognized intent type, only valid options are individual or group, received: ",
                               self.intent_type)

            elif self.robot.sensor.lower() == 'rgb' or self.robot.sensor.lower() == 'gray':
                snapshot = self.get_pixel_obs()
                prior_planes = snapshot.shape[1]*(self.num_frames-1)
                self.obs_history = np.concatenate((self.obs_history[:, -prior_planes:, :, :], snapshot), axis=1)
                ob = self.obs_history
            else:
                raise ValueError('Unknown robot sensor type')
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action, self.closed)
                      for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor.lower() == 'rgb' or self.robot.sensor.lower() == 'gray':
                snapshot = self.get_pixel_obs()
                prior_planes = snapshot.shape[1] * (self.num_frames - 1)
                self.obs_history = np.concatenate((self.obs_history[:, -prior_planes:, :, :], snapshot), axis=1)
                ob = self.obs_history
            else:
                raise ValueError('Unknown robot sensor type')
        if done:
            self.episode_info['time'] = -self.global_time * self.time_penalty / self.time_step
            self.episode_info['global_time'] = self.global_time
            info = self.episode_info  # Return full episode information at the end
        return ob, reward, done, info

    def get_pixel_obs(self):
        """  Converts an observation into a grayscale pixel array (designed for Atari network)  """
        delta = self.side/(self.pixel_side - 1)
        bd1 = -self.side/2
        bd2 = self.side/2 + delta
        x, y = np.meshgrid(np.arange(bd1, bd2, delta), np.arange(bd2, bd1, -delta))
        if self.robot.sensor.lower() == 'rgb':
            obs1, obs2, obs3 = np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape)
            # Color humans:
            for human in self.humans:
                robot_distance = np.sqrt((human.px - self.robot.px)**2 + (human.py - self.robot.py)**2)
                if robot_distance < self.robot.horizon:
                    obs1[np.nonzero((x - human.px)**2 + (y-human.py)**2 <= human.radius**2)] = 1
            # Color goal:
            obs2[np.nonzero((x - self.robot.gx) ** 2 + (y - self.robot.gy) ** 2 <= self.goal_radius ** 2)] = 1
            # Color robot:
            obs3[np.nonzero((x - self.robot.px) ** 2 + (y - self.robot.py) ** 2 <= self.robot.radius ** 2)] = 1
            obs = np.concatenate((np.expand_dims(obs1, 0), np.expand_dims(obs2, 0), np.expand_dims(obs3, 0)), axis=0)
            return np.float32(np.expand_dims(obs, 0))
        elif self.robot.sensor.lower() == 'gray':
            obs = np.zeros(x.shape)
            # Color humans:
            for human in self.humans:
                robot_distance = np.sqrt((human.px - self.robot.px) ** 2 + (human.py - self.robot.py) ** 2)
                if robot_distance < self.robot.horizon:
                    obs[np.nonzero((x - human.px) ** 2 + (y - human.py) ** 2 <= human.radius ** 2)] = 1. / 3
            # Color goal:
            obs[np.nonzero((x - self.robot.gx) ** 2 + (y - self.robot.gy) ** 2 <= self.goal_radius ** 2)] = 2. / 3
            # Color robot:
            obs[np.nonzero((x - self.robot.px) ** 2 + (y - self.robot.py) ** 2 <= self.robot.radius ** 2)] = 1.
            return np.float32(np.expand_dims(np.expand_dims(obs, 0), 0))
        else:
            raise ValueError('Robot sensor incompatible with pixel observation.')

    def render(self, mode='video'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'green'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = []

            for i in range(len(self.group_membership)):
                group_color = cmap(i)
                for _ in self.group_membership[i]:
                    human_colors.append(group_color)

            # the rest are individuals
            for i in range(len(self.individual_membership)):
                ind_color = cmap(len(self.group_membership) + i)
                human_colors.append(ind_color)

            for s in self.obstacles:
                ax.plot(s[:, 0], s[:, 1], "-o", color="black", markersize=2.5)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None',
                                 markersize=25, label='Goal')
            #goal = plt.Circle((self.robot.gx, self.robot.gy), self.goal_radius, fill=True, color=goal_color)
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)
            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=human_colors[i])
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])
            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)
            # compute attention scores
            if self.attention_weights is not None and self.robot.sensor.lower() == 'coordinates':
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]
            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None and self.robot.sensor.lower() == 'coordinates':
                        # human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if self.movie_file != '':
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(self.movie_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
