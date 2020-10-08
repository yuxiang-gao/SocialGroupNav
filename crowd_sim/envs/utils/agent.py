import numpy as np
from numpy.linalg import norm
import abc
import logging
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState

from fastdtw import fastdtw
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import euclidean

# use to store trajectory
import collections

RATE_VAL = 1  # Hz
BETA = 0.5
PLOT = 0


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.
        """
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

        self.enable_intent = config.getboolean('env', 'enable_intent', fallback=False)

        # TODO: make this configurable
        target_min_x = -6
        target_min_y = -6
        target_max_x = 6
        target_max_y = 6
        grid_length_x = 3
        grid_length_y = 3

        if self.enable_intent:

            self.targets = Agent.compute_targets_by_grid(target_min_x, target_max_x, target_min_y,
                                                         target_max_y, grid_length_x, grid_length_y)

            self.traj_length = config.getint(section, 'traj_length', fallback=8)
            self.trajectory = collections.deque(maxlen=self.traj_length)

            self.target_map = np.ones(grid_length_x*grid_length_y)/(grid_length_x*grid_length_y)

        else:
            self.target_map = None


    @staticmethod
    def compute_targets_by_grid(min_x=-20, max_x=20, min_y=-20, max_y=20, grid_length_x=4, grid_length_y=4):
        total_distance_x = max_x - min_x
        distance_per_bin_x = total_distance_x / grid_length_x
        center_offset_x = distance_per_bin_x / 2

        total_distance_y = max_y - min_y
        distance_per_bin_y = total_distance_y / grid_length_y
        center_offset_y = distance_per_bin_y / 2

        all_targets = []

        for i in range(0, grid_length_x):
            for j in range(0, grid_length_y):
                x = distance_per_bin_x * i + center_offset_x - total_distance_x / 2
                y = distance_per_bin_y * j + center_offset_y - total_distance_y / 2

                all_targets.append([x, y])

        return np.array(all_targets)

    @staticmethod
    def get_trajectory(start_pos, end_pos, speed, rate, num_samples):
        # compute the number of samples based on speed
        distance = np.linalg.norm(end_pos - start_pos)

        if speed == 0:
            samples = 8
        else:
            time = distance / speed
            samples = int(round(time / rate))

        if samples > 8:
            samples = 8

        x = np.linspace(start_pos[0], end_pos[0], num=samples)
        y = np.linspace(start_pos[1], end_pos[1], num=samples)

        z = np.concatenate((x, y), 0)
        full_traj = z.reshape((2, samples))
        full_traj = full_traj.transpose()

        if samples >= num_samples:
            return full_traj[0:num_samples, :]
        else:
            return full_traj

    @staticmethod
    def compute_target_map(traj, obs_len, targets):

        start_idx = 0
        end_idx = obs_len

        this_sample = traj[start_idx:end_idx, :]
        start_pos = this_sample[0]

        end_pos = this_sample[len(this_sample) - 1]

        # estimate the speed
        distance_travelled = np.linalg.norm(end_pos - start_pos)
        speed = distance_travelled / (1 / RATE_VAL * obs_len)

        traj_to_targets = []

        p_h_g = []

        #import time
        #start = time.time()

        for i in range(len(targets)):
            traj = Agent.get_trajectory(start_pos, targets[i], speed, RATE_VAL, obs_len)

            traj_to_targets.append(traj)
            # print (traj)
            # print ('\n')
            try:

                dist, path = fastdtw(this_sample, traj, dist=euclidean)


            except:
                print('boo')
            # print (dist)
            # if (dist < shortest_dist):
            # 	shortest_dist = dist
            # 	shortest_dist_idx = i

            prob = math.exp(-BETA * dist)
            p_h_g.append(prob)

        #end = time.time()
        #print(end - start)

        prob_sum = sum(p_h_g)
        p_h_g = [x / prob_sum for x in p_h_g]

        # targets[0] = p_h_g[0]

        # idx = p_h_g.index(max(p_h_g))
        # print (targets[idx])
        # print (target_pos)
        # closest_target_idx, closest_target = find_nearest(targets, target_pos)
        # print (closest_target_idx)

        if PLOT:
            print(p_h_g)
            plt.scatter(this_sample[:, 0], this_sample[:, 1], c='red')
            targets_np = np.array(targets)

            for d in range(len(traj_to_targets)):
                plt.scatter(traj_to_targets[d][:, 0], traj_to_targets[d][:, 1], c='green', alpha=p_h_g[d]);
                plt.scatter(targets_np[d, 0], targets_np[d, 1], c='black', s=200, alpha=p_h_g[d]);

            plt.show()

        return p_h_g

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)  # .2, .8

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

        if self.enable_intent:
            self.append_trajectory((px, py))

    def update_target_map(self):
        curr_ped_seq = np.array(list(self.trajectory))
        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
        rel_curr_ped_seq[:, 1:] = \
            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

        self.target_map = Agent.compute_target_map(rel_curr_ped_seq, self.traj_length, self.targets)

    def append_trajectory(self, p):
        self.trajectory.append(p)
        self.update_target_map()

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius, self.target_map)

    def get_trajectory_history(self):
        return self.trajectory

    def get_target_map(self):
        return self.target_map

    def get_next_observable_state(self, action, closed=True):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step, closed)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)

        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius, self.target_map)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.target_map)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

        if self.enable_intent:
            self.append_trajectory((self.px, self.py))

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob, groups=None):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t, closed=True, x_min=-6, x_max=6, y_min=-6, y_max=6):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t
        if closed:
            px, py = self.reflect(px, py, x_min, x_max, y_min, y_max)
        return px, py

    def step(self, action, closed=True):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step, closed)
        self.px, self.py = pos

        if self.enable_intent:
            self.append_trajectory((self.px, self.py))

        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

    @staticmethod
    def reflect(px, py, x_min, x_max, y_min, y_max):
        """  Reflects objects to keep them in pre-defined box.  """
        if x_min > px:
            px = x_min + (x_min - px)
        if x_max < px:
            px = x_max - (px - x_max)
        if y_min > py:
            py = y_min + (y_min - py)
        if y_max < py:
            py = y_max - (py - y_max)
        return px, py
