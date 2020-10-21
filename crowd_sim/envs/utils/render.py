import logging

import numpy as np

import pygame
from pygame.locals import *
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.action import ActionXY, ActionRot

cmap = plt.cm.get_cmap("tab20")
cmap2 = plt.cm.get_cmap("Set1")
robot_color = cmap2(1)
goal_color = cmap2(2)
arrow_color = cmap2(0)

WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
BLACK = pygame.Color(0, 0, 0)
MOVE_BINDINGS = {
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
SPEED_BINDINGS = {
    "q": (1.1, 1.1),
    "z": (0.9, 0.9),
    "w": (1.1, 1),
    "x": (0.9, 1),
    "e": (1, 1.1),
    "c": (1, 0.9),
}

KEY_LOOKUP = {
    "escape": pygame.K_ESCAPE,
    "space": pygame.K_SPACE,
    ",": pygame.K_COMMA,
    "minus": pygame.K_MINUS,
    ".": pygame.K_PERIOD,
    "semicolon": pygame.K_SEMICOLON,
    "less-than": pygame.K_LESS,
    "equals": pygame.K_EQUALS,
    "greater-than": pygame.K_GREATER,
    "lbracket": pygame.K_LEFTBRACKET,
    "rbracket": pygame.K_RIGHTBRACKET,
    "backslash": pygame.K_BACKSLASH,
    "caret": pygame.K_CARET,
    "underscore": pygame.K_UNDERSCORE,
    "grave": pygame.K_BACKQUOTE,
    "euro": pygame.K_EURO,
    "a": pygame.K_a,
    "b": pygame.K_b,
    "c": pygame.K_c,
    "d": pygame.K_d,
    "e": pygame.K_e,
    "f": pygame.K_f,
    "g": pygame.K_g,
    "h": pygame.K_h,
    "i": pygame.K_i,
    "j": pygame.K_j,
    "k": pygame.K_k,
    "l": pygame.K_l,
    "m": pygame.K_m,
    "n": pygame.K_n,
    "o": pygame.K_o,
    "p": pygame.K_p,
    "q": pygame.K_q,
    "r": pygame.K_r,
    "s": pygame.K_s,
    "t": pygame.K_t,
    "u": pygame.K_u,
    "v": pygame.K_v,
    "w": pygame.K_w,
    "x": pygame.K_x,
    "y": pygame.K_y,
    "z": pygame.K_z,
    "0": pygame.K_0,
    "1": pygame.K_1,
    "2": pygame.K_2,
    "3": pygame.K_3,
    "4": pygame.K_4,
    "5": pygame.K_5,
    "6": pygame.K_6,
    "7": pygame.K_7,
    "8": pygame.K_8,
    "9": pygame.K_9,
    "kp_period": pygame.K_KP_PERIOD,
    "kp_divide": pygame.K_KP_DIVIDE,
    "kp_multiply": pygame.K_KP_MULTIPLY,
    "kp_minus": pygame.K_KP_MINUS,
    "kp_plus": pygame.K_KP_PLUS,
    "kp_enter": pygame.K_KP_ENTER,
    "kp_equals": pygame.K_KP_EQUALS,
    "up": pygame.K_UP,
    "down": pygame.K_DOWN,
    "right": pygame.K_RIGHT,
    "left": pygame.K_LEFT,
    "insert": pygame.K_INSERT,
    "delete": pygame.K_DELETE,
    "home": pygame.K_HOME,
    "end": pygame.K_END,
    "pageup": pygame.K_PAGEUP,
    "pagedown": pygame.K_PAGEDOWN,
    "F1": pygame.K_F1,
    "F2": pygame.K_F2,
    "F3": pygame.K_F3,
    "F4": pygame.K_F4,
    "F5": pygame.K_F5,
    "F6": pygame.K_F6,
    "F7": pygame.K_F7,
    "F8": pygame.K_F8,
    "F9": pygame.K_F9,
    "F10": pygame.K_F10,
    "F11": pygame.K_F11,
    "F12": pygame.K_F12,
    "F13": pygame.K_F13,
    "F14": pygame.K_F14,
    "F15": pygame.K_F15,
    "numlock": pygame.K_NUMLOCK,
    "capslock": pygame.K_CAPSLOCK,
    "scrollock": pygame.K_SCROLLOCK,
    "shift": pygame.K_LSHIFT,
    "lshift": pygame.K_LSHIFT,
    "rshift": pygame.K_RSHIFT,
    "ctrl": pygame.K_LCTRL,
    "lctrl": pygame.K_LCTRL,
    "rctrl": pygame.K_RCTRL,
    "alt": pygame.K_LALT,
    "lalt": pygame.K_LALT,
    "ralt": pygame.K_RALT,
    "meta": pygame.K_LMETA,
    "lmeta": pygame.K_LMETA,
    "rmeta": pygame.K_RMETA,
    "windows": pygame.K_LSUPER,
    "lwindows": pygame.K_LSUPER,
    "rwindows": pygame.K_RSUPER,
    "shift": pygame.K_MODE,
    "help": pygame.K_HELP,
    "screen": pygame.K_PRINT,
    "sysrq": pygame.K_SYSREQ,
    "break": pygame.K_BREAK,
    "menu": pygame.K_MENU,
    "power": pygame.K_POWER,
}


def lookup_cmd(key_down):
    for key, value in KEY_LOOKUP.items():
        if key_down == value:
            logging.info(f"Key {key} pushed.")
            if key in MOVE_BINDINGS.keys():
                return "move", MOVE_BINDINGS.get(key)
            elif key in SPEED_BINDINGS.keys():
                return "speed", SPEED_BINDINGS.get(key)
    return None, None


def lookup_cmd_dict(key_input):
    for key, value in KEY_LOOKUP.items():
        if key_input[value]:
            logging.info(f"Key {key} pushed.")
            if key in MOVE_BINDINGS.keys():
                return "move", MOVE_BINDINGS.get(key)
            elif key in SPEED_BINDINGS.keys():
                return "speed", SPEED_BINDINGS.get(key)
    return None, None


class GameAgent(pygame.sprite.Sprite):
    def __init__(
        self,
        pos=(0, 0),
        angle=0,
        radius=10,
        color=BLACK,
        image="crowd_sim/envs/utils/resources/ped_avatar.png",
    ):
        super().__init__()
        self.original_image = pygame.image.load(image)
        self.original_image = pygame.transform.rotate(self.original_image, -90)
        self.original_image = pygame.transform.scale(self.original_image, (2 * radius, 2 * radius))

        # pygame.draw.circle(self.original_image, color, (radius, radius), radius, 3)
        pygame.draw.rect(self.original_image, color, (0, 0, 2 * radius, 2 * radius), 2)
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.pos = pos
        self.angle = angle

    def update(self, pos, angle):
        self.pos = pos
        self.angle = angle
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = pos

    def draw(self, screen):
        screen.blit(self.image, self.rect)


class GameHuman(GameAgent):
    def __init__(self, pos, color, id, radius=10):
        super().__init__(pos=pos, color=color)
        self.id = id


class GameRobot(GameAgent):
    def __init__(self, pos, color, radius=10):
        super().__init__(pos=pos, color=color, image="crowd_sim/envs/utils/resources/robot.png")
        self.speed = 2
        self.turn = 10
        self.moving = False

    def handle_input(self, cmd):
        # key_input = pygame.key.get_pressed()
        # _, cmd = lookup_cmd_dict(key_input)
        vx, vy = 0, 0
        if len(cmd) == 4:
            x, _, _, th = cmd
            th = self.angle + th * self.turn
            vx = np.cos(np.deg2rad(th)) * x
            vy = np.sin(np.deg2rad(th)) * x
            vx *= self.speed
            vy *= self.speed
            self.update((self.pos[0] + vx, self.pos[1] + vy), th)
            # print(th, vx, vy)
        elif len(cmd) == 2:
            self.speed *= cmd[0]
            self.turn *= cmd[1]

        action = ActionXY(vx, vy)

        return action


# %%
class App:
    def __init__(self, env):
        self.clock = pygame.time.Clock()
        self.env = env
        self._running = True
        self._display = None
        self.size = np.array((640, 640))

        self.sim_size = np.array(self.env.scene_manager.scenario_config.map_size)
        self.scale = self.size / 2 / self.sim_size

        self.robot = None
        self.humans = []

        # generate color mapping
        self.human_colors = [0] * len(self.env.humans)
        for i in range(len(self.env.group_membership)):
            group_color = App.convert_color(cmap(i))
            for idx in self.env.group_membership[i]:
                self.human_colors[idx] = group_color

        for idx in self.env.individual_membership:
            ind_color = App.convert_color(cmap(len(self.env.group_membership) + idx))
            self.human_colors[idx] = ind_color

    def sim_to_canvas(self, coord):
        coord = np.asarray(coord)
        coord = coord * self.scale
        coord *= np.array((1, -1))
        coord += self.size / 2
        return coord

    def canvas_to_sim(self, coord):
        coord -= self.size / 2
        coord *= np.array((1, -1))
        coord = coord / self.scale
        return coord

    def on_init(self):
        pygame.init()
        self._display = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Social Group Navigation")
        self._running = True

        # self.player = Player(
        #     pos=self.sim_to_canvas(self.env.robot.get_start_position()), color=robot_color
        # )
        self.robot = GameRobot(
            self.sim_to_canvas(self.env.robot.get_start_position()),
            App.convert_color(robot_color),
            radius=self.env.robot.radius * self.scale,
        )

        for i, human in enumerate(self.env.states[-1][1]):
            self.humans.append(
                GameHuman(
                    self.sim_to_canvas(human.position),
                    self.human_colors[i],
                    i,
                    radius=self.env.humans[0].radius * self.scale,
                )
            )

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                logging.info("Terminating.")
                self._running = False

            # elif event.type == MOUSEBUTTONDOWN:
            #     if self.robot.rect.collidepoint(event.pos):
            #         self.robot.moving = True

            # elif event.type == MOUSEBUTTONUP:
            #     self.robot.moving = False

            # elif event.type == MOUSEMOTION and self.robot.moving:
            #     self.robot.rect.move_ip(event.rel)
            else:
                cmd_type, cmd = lookup_cmd(event.key)
                if cmd is not None:
                    action = self.robot.handle_input(cmd)
                    # logging.info(f"{cmd_type} cmd: {cmd}.")
                    return action
        return None

    def on_loop(self):

        pass

    def on_render(self):
        self._display.fill(WHITE)
        for ob in self.env.obstacles:
            ob = np.asarray(ob)
            pygame.draw.line(
                self._display,
                BLACK,
                self.sim_to_canvas(ob[[0, 2]]),
                self.sim_to_canvas(ob[[1, 3]]),
                3,
            )

        for i in range(len(self.env.humans)):
            human = self.env.humans[i]
            pygame.draw.rect(
                self._display,
                self.human_colors[i],
                (*self.sim_to_canvas(human.get_goal_position()), 10, 10),
                3,
            )
            pygame.draw.circle(
                self._display,
                self.human_colors[i],
                np.int0(self.sim_to_canvas(human.get_start_position())),
                10,
                3,
            )

        for i, human in enumerate(self.env.states[-1][1]):
            self.angle = np.rad2deg(np.arctan2(human.vy, human.vx))
            self.humans[i].update(self.sim_to_canvas(human.position), self.angle)
            self.humans[i].draw(self._display)
            # print(self.humans[i].id, self.humans[i].angle)

        self.robot.update(self.sim_to_canvas(self.env.robot.get_position()), self.robot.angle)
        self.robot.draw(self._display)
        # pygame.display.flip()
        pygame.display.update()
        self.clock.tick(40)

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def step(self):
        action = None
        for event in pygame.event.get():
            action = self.on_event(event)
        self.on_loop()
        self.on_render()
        self.clock.tick(40)
        return action if action is not None else ActionXY(0, 0)

    @staticmethod
    def convert_color(mat_color):
        return np.int0(np.array(mat_color) * 255)


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    theApp = App(0)
    theApp.on_execute()
