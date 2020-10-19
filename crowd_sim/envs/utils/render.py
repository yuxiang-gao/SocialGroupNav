import logging

import numpy as np

import pygame
from pygame.locals import *
import matplotlib.pyplot as plt

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


class Player:
    def __init__(self, pos, color):
        self.rect = pygame.rect.Rect((*pos, 10, 10))
        self.color = color

    def handle_input(self):
        key = pygame.key.get_pressed()
        dist = 1
        if key[pygame.K_LEFT]:
            self.rect.move_ip(-1, 0)
        if key[pygame.K_RIGHT]:
            self.rect.move_ip(1, 0)
        if key[pygame.K_UP]:
            self.rect.move_ip(0, -1)
        if key[pygame.K_DOWN]:
            self.rect.move_ip(0, 1)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)


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

        self.player = None

        # generate color mapping
        self.human_colors = [0] * len(self.env.humans)
        for i in range(len(self.env.group_membership)):
            group_color = App.convert_color(cmap(i))
            for idx in self.env.group_membership[i]:
                self.human_colors[idx] = group_color

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

        self.player = Player(self.sim_to_canvas(self.env.robot.get_start_position()), robot_color)

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                logging.info("Terminating.")
                self._running = False
            else:
                cmd_type, cmd = lookup_cmd(event.key)
                logging.info(f"{cmd_type} cmd: {cmd}.")

    def on_loop(self):
        self.player.handle_input()

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
        self.player.draw(self._display)
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

    @staticmethod
    def convert_color(mat_color):
        return np.int0(np.array(mat_color) * 255)


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    theApp = App(0)
    theApp.on_execute()
