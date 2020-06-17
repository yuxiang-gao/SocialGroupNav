import numpy as np
from pysocialforce import Simulator
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class SocialForce(Policy):
    def __init__(self):
        super().__init__()
        self.name = "SocialForce"
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = "holonomic"
        self.sim = None

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """

        :param state:
        :return:
        """
        sf_state = []
        robot_state = state.robot_state
        sf_state.append(
            (
                robot_state.px,
                robot_state.py,
                robot_state.vx,
                robot_state.vy,
                robot_state.gx,
                robot_state.gy,
            )
        )
        for human_state in state.human_states:
            # approximate desired direction with current velocity
            if human_state.vx == 0 and human_state.vy == 0:
                gx = np.random.random()
                gy = np.random.random()
            else:
                gx = human_state.px + human_state.vx
                gy = human_state.py + human_state.vy
            sf_state.append(
                (human_state.px, human_state.py, human_state.vx, human_state.vy, gx, gy)
            )
        groups = None  # add group info here
        sim = Simulator(np.array(sf_state), groups=groups)
        sim.step()
        action = ActionXY(sim.state[0, 2], sim.state[0, 3])

        self.last_state = state

        return action


class CentralizedSocialForce(SocialForce):
    """
    Centralized socialforce, a bit different from decentralized socialforce, where the goal position of other agents is
    set to be (0, 0)
    """

    def __init__(self):
        super().__init__()

    def predict(self, state):
        sf_state = []
        for agent_state in state:
            sf_state.append(
                (
                    agent_state.px,
                    agent_state.py,
                    agent_state.vx,
                    agent_state.vy,
                    agent_state.gx,
                    agent_state.gy,
                )
            )
        groups = None
        sim = Simulator(np.array(sf_state), groups=groups)
        sim.step()
        actions = [ActionXY(sim.state[i, 2], sim.state[i, 3]) for i in range(len(state))]
        del sim

        return actions
