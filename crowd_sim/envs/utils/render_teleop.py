import numpy as np
import sys
import pygame
from pygame.locals import *
from crowd_sim.envs.utils.humans import Human

class pyg_render(center_x, center_y, width, height, humans, h_dir):#Input initial center of robot, width and height of canvas and a list of human positions and directions
    #Initializing the robot starting position and canvas size
    def _init_(self, center_x, center_y, width, height):
        #for canvas
        self.width = width
        self.height = height
        #for robot
        pos = (center_x, center_y)
        speed = [0, 0]
        state = 0
        #colors
        WHITE = [255, 255, 255]
        BLACK = [0, 0, 0]
        RED = [255, 0, 0]
        GREEN = [0, 255, 0]
        BLUE = [0, 0, 255]
        #human
        human_pos = humans
        human_dir = h_dir

    def run():
        pygame.init()
        screen = pygame.display.set_mode(size)
        bot = pygame.Surface((20, 20))
        rect = bot.get_rect(center = pos)
        bot.fill(BLUE)

        while 1:
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
            center_rect = (rect.centerx, rect.centery)
    
            pygame.draw.line(screen, BLACK, center_rect, (np.array(center_rect)+ (20* np.array(speed))), 3)
            if elapsed % 30 == 0:
        
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

            for human in human_pos:
                for dir in human_dir:
                pygame.draw.circle(screen, RED, (human[0], human[1]), 10)
                pygame.draw.line(screen, BLACK, d1, (np.array(human) + (10* np.array(dir))), 3)
                

            
        return 0
