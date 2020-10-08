class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, target_map=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

        if target_map is None:
            self.add_intent = False
        else:
            self.add_intent = True

        if self.add_intent:

            self.target_map_0 = target_map[0]
            self.target_map_1 = target_map[1]
            self.target_map_2 = target_map[2]
            self.target_map_3 = target_map[3]

            self.target_map_4 = target_map[4]
            self.target_map_5 = target_map[5]
            self.target_map_6 = target_map[6]
            self.target_map_7 = target_map[7]

            self.target_map_8 = target_map[8]
            # self.target_map_9 = target_map[9]
            # self.target_map_10 = target_map[10]
            # self.target_map_11 = target_map[11]
            #
            # self.target_map_12 = target_map[12]
            # self.target_map_13 = target_map[13]
            # self.target_map_14 = target_map[14]
            # self.target_map_15 = target_map[15]

    def update_target_map(self, target_map):

        if self.add_intent:
            self.target_map_0 = target_map[0]
            self.target_map_1 = target_map[1]
            self.target_map_2 = target_map[2]
            self.target_map_3 = target_map[3]

            self.target_map_4 = target_map[4]
            self.target_map_5 = target_map[5]
            self.target_map_6 = target_map[6]
            self.target_map_7 = target_map[7]

            self.target_map_8 = target_map[8]

    def __add__(self, other):
        if self.add_intent:
            return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,
                            self.target_map_0, self.target_map_1, self.target_map_2,
                            self.target_map_3, self.target_map_4, self.target_map_5,
                            self.target_map_6, self.target_map_7, self.target_map_8)
        else:
            return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        if self.add_intent:
            return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                              self.v_pref, self.theta,
                            self.target_map_0, self.target_map_1, self.target_map_2,
                            self.target_map_3, self.target_map_4, self.target_map_5,
                            self.target_map_6, self.target_map_7, self.target_map_8]])
        else:
            return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                              self.v_pref, self.theta]])


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius, target_map=None):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

        if target_map is None:
            self.add_intent = False
        else:
            self.add_intent = True

        if self.add_intent:
            self.target_map_0 = target_map[0]
            self.target_map_1 = target_map[1]
            self.target_map_2 = target_map[2]
            self.target_map_3 = target_map[3]

            self.target_map_4 = target_map[4]
            self.target_map_5 = target_map[5]
            self.target_map_6 = target_map[6]
            self.target_map_7 = target_map[7]

            self.target_map_8 = target_map[8]
            # self.target_map_9 = target_map[9]
            # self.target_map_10 = target_map[10]
            # self.target_map_11 = target_map[11]
            #
            # self.target_map_12 = target_map[12]
            # self.target_map_13 = target_map[13]
            # self.target_map_14 = target_map[14]
            # self.target_map_15 = target_map[15]

    def update_target_map(self, target_map):
        if self.add_intent:
            self.target_map_0 = target_map[0]
            self.target_map_1 = target_map[1]
            self.target_map_2 = target_map[2]
            self.target_map_3 = target_map[3]

            self.target_map_4 = target_map[4]
            self.target_map_5 = target_map[5]
            self.target_map_6 = target_map[6]
            self.target_map_7 = target_map[7]

            self.target_map_8 = target_map[8]

    def __add__(self, other):
        if self.add_intent:
            return other + (self.px, self.py, self.vx, self.vy, self.radius,
                            self.target_map_0, self.target_map_1, self.target_map_2,
                            self.target_map_3, self.target_map_4, self.target_map_5,
                            self.target_map_6, self.target_map_7, self.target_map_8)

        else:
            return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        if self.add_intent:
            return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius,
                            self.target_map_0, self.target_map_1, self.target_map_2,
                            self.target_map_3, self.target_map_4, self.target_map_5,
                            self.target_map_6, self.target_map_7, self.target_map_8]])

        else:
            return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
           assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
