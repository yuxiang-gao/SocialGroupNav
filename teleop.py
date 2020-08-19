import os
import numpy as np
import math
from numpy.linalg import norm
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
from matplotlib import patches

matplotlib.use("TKAgg")

# plt.rcParams["animation.html"] = "jshtml"
# from IPython.display import HTML

import gym
import importlib.util

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_nav.utils.explorer import Explorer
from crowd_nav.configs.icra_benchmark.config import (
    BaseEnvConfig,
    BasePolicyConfig,
    BaseTrainConfig,
    Config,
)

import logging

import threading

import sys, select, termios, tty

import numpy as np

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY, ActionRot

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .
For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >
t : up (+z)
b : down (-z)
anything else : stop
q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
CTRL-C to quit
"""

moveBindings = {
    "i": (1, 0, 0, 0),
    "o": (1, 0, 0, -1),
    "j": (0, 0, 0, 1),
    "l": (0, 0, 0, -1),
    "u": (1, 0, 0, 1),
    ",": (-1, 0, 0, 0),
    ".": (-1, 0, 0, 1),
    "m": (-1, 0, 0, -1),
    "O": (1, -1, 0, 0),
    "I": (1, 0, 0, 0),
    "J": (0, 1, 0, 0),
    "L": (0, -1, 0, 0),
    "U": (1, 1, 0, 0),
    "<": (-1, 0, 0, 0),
    ">": (-1, -1, 0, 0),
    "M": (-1, 1, 0, 0),
    "t": (0, 0, 1, 0),
    "b": (0, 0, -1, 0),
}

speedBindings = {
    "q": (1.1, 1.1),
    "z": (0.9, 0.9),
    "w": (1.1, 1),
    "x": (0.9, 1),
    "e": (1, 1.1),
    "c": (1, 0.9),
}


def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed, turn)


def getKey(key_timeout, settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        # self.publisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.twist = np.zeros(6)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def update(self, x, y, z, th, speed, turn):
        self.condition.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()

    def run(self):
        while not self.done:
            self.condition.acquire()
            # Wait for a new message or timeout.
            self.condition.wait(self.timeout)

            # Copy state into twist message.
            self.twist = np.asarray(
                [
                    self.x * self.speed,
                    self.y * self.speed,
                    self.z * self.speed,
                    0,
                    0,
                    self.th * self.turn,
                ]
            )

            self.condition.release()

        # Publish stop message when thread exits.
        self.twist = np.zeros(6)


env = gym.make("CrowdSim-v0")


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.playback = False
        self.dataset = "./data/sgan-datasets/eth/train/biwi_hotel_train.txt"
        self.env.randomize_attributes = True
        self.sim.human_num = 8
        self.sim.centralized_planning = False
        self.humans.policy = "socialforce"


config = EnvConfig(debug=True)

env.configure(config)
robot = Robot(config, "robot")
robot.timestep = 0.25
robot.policy = policy_factory["socialforce"]()
robot.kinematics = "holonomic"
env.set_robot(robot)

rewards = []
ob = env.reset()
last_pos = np.array(robot.get_position())
done = False
# while not done:
#     action = robot.act(ob)
#     ob, _, done, info = env.step(action)
#     rewards.append(_)
#     current_pos = np.array(robot.get_position())
#     print(f"Speed: {np.linalg.norm(current_pos - last_pos) / robot.time_step}")
#     last_pos = current_pos
#     env.render("traj")


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    speed = 2
    turn = 10
    repeat = 0.0
    key_timeout = 0.0
    if key_timeout == 0.0:
        key_timeout = None
    pub_thread = PublishThread(repeat)

    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    states = None

    try:
        pub_thread.update(x, y, z, th, speed, turn)
        print(msg)
        print(vels(speed, turn))

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x(m)", fontsize=16)
        ax.set_ylabel("y(m)", fontsize=16)
        # plt.ion()
        # fig.show()
        fig.canvas.draw()
        while not done:
            action = robot.act(ob)
            state = robot.get_full_state()
            theta = robot.theta + np.deg2rad(pub_thread.th * pub_thread.turn)
            robot.theta = theta  # ActionXY does not have member v for non-holonomic agents, manually update theta now
            x = pub_thread.x
            y = pub_thread.y
            vx = np.cos(theta) * x + np.sin(theta) * y
            vy = np.cos(theta) * y + np.sin(theta) * x
            vx *= pub_thread.speed
            vy *= pub_thread.speed
            action = ActionXY(vx, vy)

            # action = robot.act(ob)
            ob, _, done, info = env.step(action)
            rewards.append(_)
            current_pos = np.array(robot.get_position())
            print(
                f"Pos: {current_pos} Speed: {np.linalg.norm(current_pos - last_pos) / robot.time_step}"
            )
            last_pos = current_pos
            states = env.render("teleop", plot=(fig, ax))

            key = getKey(key_timeout, settings)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]

                print(vels(speed, turn))
                if status == 14:
                    print(msg)
                status = (status + 1) % 15
            elif key == "s":
                print("Save and quit.")
                from pickle import dump

                with open("states.pkl", "wb") as f:
                    dump(states, f)
                break
            else:
                # Skip updating cmd_vel if key timeout and robot already
                # stopped.
                if key == "" and x == 0 and y == 0 and z == 0 and th == 0:
                    continue
                x = 0
                y = 0
                z = 0
                th = 0
                if key == "\x03":
                    break

            pub_thread.update(x, y, z, th, speed, turn)

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()
        plt.close()
        # print(states)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
