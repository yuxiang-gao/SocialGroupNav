import numpy as np
import math
import sys
import pygame
from pygame.locals import *
from sympy.solvers import solve
from sympy import Symbol

pygame.init()
size = width, height = 600, 600
speed = [2, 2]
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0 ,0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]

screen = pygame.display.set_mode(size)

bot = pygame.Surface((20, 20))
robot_positions = [self.states[i][0].position for i in range(len(self.states))]
human_positions = [
    [self.states[i][1][j].position for j in range(len(self.humans))]
    for i in range(len(self.states))
]
rect = bot.get_rect(center = (robot_positions[0][0]))
bot.fill(BLUE)
state = 0

speed = np.zeros(2)

def add(t1, t2):
    t_x = t1[0] + t2[0]
    t_y = t1[1] + t2[1]
    return (t_x, t_y)

def mul(t1, t2):
    t_x = t1[0] * t2[0]
    t_y = t1[1] * t2[1]
    return (t_x, t_y)

def get_line_function(p1, p2):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - k * p2[0]
    return (k, b)

def dist(p1, p2):
    distance = math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
    return distance

def tri(p1, p2):
    k, b = get_line_function(p1, p2)
    x = Symbol('x')
    x1, x2 = solve(((p2[0] - x) ** 2) + ((p2[1] - (k * x + b)) ** 2) - 100, x)
    y1 = x1 * k + b
    y2 = x2 * k + b
    
    if (dist((x1, y1), p2) < dist((x2, y2), p2)):
        tri1 = (x1, y1)
    else:
        tri1 = (x2, y2)

    # calculate perpendicular line and get two intercepts
    #b_p = y3 + (1 / k) * x3
#global_time = k * self.time_step    
while 1:
    screen.fill(WHITE)
    elapsed = pygame.time.get_ticks()
    #Moving around:
    #   u    i    o
    #   j    k    l
    #   m    ,    .    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_i:
                state = 1
                speed[0] = 0
                speed[1] = -1
            if event.key == K_COMMA:
                state = 2
                speed[0] = 0
                speed[1] = 1
            if event.key == K_j:
                state = 3
                speed[0] = -1
                speed[1] = 0
            if event.key == K_l:
                state = 4
                speed[0] = 1
                speed[1] = 0
            if event.key == K_u:
                state = 5
                speed[0] = -1
                speed[1] = -1
            if event.key == K_o:
                state = 6
                speed[0] = 1
                speed[1] = -1
            if event.key == K_m:
                state = 7
                speed[0] = -1
                speed[1] = 1
            if event.key == K_PERIOD:
                state = 8
                speed[0] = 1
                speed[1] = 1
            if event.key == K_k:
                state = 0
                speed[0] = 0
                speed[1] = 0   
    human_colors = [cmap(i) for i in range(len(self.humans))]
    #for k in range(len(self.states)):
    k = pygame.time.get_ticks()
    if k % 4 == 0 or k == len(self.states) - 1:
        rect = robot_positions[-1]
        pygame.draw.line(screen, BLACK, center_rect, add(center_rect, (20* np.array(speed))), 3)
        rect = bot.get_rect(center = robot_positions[k])
        bot.fill(BLUE)
        rect.top += speed[1]
        rect.left += speed[0]
        if rect.top < 0:
            rect.top = 0
        elif rect.bottom > 600:
            rect.bottom = 600
        elif rect.left < 0:
            rect.left = 0
        elif rect.right > 600:
            rect.right = 600

        for i in range(len(self.humans)):
                
            color = human_colors[i]
            human = self.humans[i]
            human_direction = (self.states[k - 1][1][i].px, self.states[k][1][i].py)
                
            pygame.draw.circle(screen, color, human_positions[j][i], 10)
            pygame.draw.line(screen, BLACK, human_positions[j][i], add(d, (10* np.array(human_direction))), 3)   
    
    screen.blit(bot, rect)
    pygame.display.update()
