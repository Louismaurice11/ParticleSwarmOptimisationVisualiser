################################### Imports ########################################
import pygame
import numpy as np
from Main import *

################################### Map Class ########################################
class Map:
    # Initialize the map
    def __init__(self, grid_size, testMap):
        global SCREEN
        global WINDOW_SIZE
        global GRID_SIZE
        SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.grid_size = grid_size
        self.oGrid = [[0] * grid_size for _ in range(grid_size)]
        self.obstacles = []
        self.line_width = 1
        self.obstacle_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        
        self.obstacle_surface.fill((255, 255, 255))
        if testMap == 0:
            self.load_from_file('grid.txt')
        elif testMap == 1:
            self.load_from_file('grid1.txt')
        elif testMap == 2:
            self.load_from_file('grid2.txt')
        elif testMap == 3:
            self.load_from_file('grid3.txt')
        self.draw_obstacles_once()
    
    # Draw the obstacles on the map
    def draw_obstacles_once(self):      
        self.obstacle_surface.fill((0, 0, 0, 0))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.oGrid[i][j] == 1:
                    color = (0, 0, 0)
                    pygame.draw.rect(self.obstacle_surface, color, 
                                     (i * (WINDOW_SIZE // self.grid_size), 
                                      j * (WINDOW_SIZE // self.grid_size), 
                                      WINDOW_SIZE // self.grid_size, 
                                      WINDOW_SIZE // self.grid_size))
                    
    # Draw the map
    def draw(self):
        SCREEN.blit(self.obstacle_surface, (0, 0))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.oGrid[x][y] == 2:
                    pygame.draw.circle(SCREEN, (0, 0, 255), 
                                       (x * (WINDOW_SIZE // self.grid_size) + WINDOW_SIZE // (2 * self.grid_size), 
                                        y * (WINDOW_SIZE // self.grid_size) + WINDOW_SIZE // (2 * self.grid_size)), 
                                        10)
                if self.oGrid[x][y] == 3:
                    pygame.draw.circle(SCREEN, (255, 0, 255), 
                                       (x * (WINDOW_SIZE // self.grid_size) + WINDOW_SIZE // (2 * self.grid_size), 
                                        y * (WINDOW_SIZE // self.grid_size) + WINDOW_SIZE // (2 * self.grid_size)), 
                                        10)
                    
    # Load the map from a file
    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                for i, line in enumerate(file):
                    self.oGrid[i] = list(map(int, line.strip().split(' ')))
            self.draw_obstacles_once()  
            print("File loaded successfully")
        except Exception as e:
            print(f"Error loading from file: {e}")

    # Load the spawn position of the swarm
    def loadSpawnPos(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.oGrid[i][j] == 3:
                    return (i, j)
        return None
    
    # Load the target positions
    def loadTargetPositions(self):
        Targets = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.oGrid[i][j] == 2:
                    Targets.append((i, j))
        return Targets
    
    # Place an obstacle on the map
    def place_obstacle(self, x, y, width, o, length):
        if o == 0:
            for i in range(x - length, x + length):
                for j in range(y - width, y + width):
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        self.oGrid[i][j] = 1
        else:
            for i in range(x - width, x + width):
                for j in range(y - length, y + length):
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        self.oGrid[i][j] = 1
        self.draw_obstacles_once()
    
    # Save the map to a file
    def save_to_file(self, filename):
        print(f"Saving to file: {filename}")  
        with open(filename, 'w') as file:
            for row in self.oGrid:
                file.write(' '.join(map(str, row)) + '\n')
        print("File saved successfully")  





################################### BuildMap Function ########################################
def BuildMap():
    global GRID_SIZE
    map = Map(GRID_SIZE, 0)
    print("CONTROLS: \n")
    print("Left Click: Place Obstacle")
    print("Right Click: Remove Obstacle")
    print("t: Place Target")
    print("q: Place Swarm")
    print("s: Save Map")
    print("o: Change orientation of obstacles")
    print("+/-: Increase/Decrease size of obstacles\n")
    placeTarget = False
    placeSwarm = False
    running = True
    iCount = 0
    o = 0
    while running:        
        iCount += 1
        SCREEN.fill((255, 255, 255))
        map.draw()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        target_x = mouse_x // (WINDOW_SIZE // GRID_SIZE)
        target_y = mouse_y // (WINDOW_SIZE // GRID_SIZE)
        length = 400
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if placeSwarm:
                    if event.button == 1:
                        if 0 <= target_x < GRID_SIZE and 0 <= target_y < GRID_SIZE:
                            map.oGrid[target_x][target_y] = 3
                    if event.button == 3:
                        if 0 <= target_x < GRID_SIZE and 0 <= target_y < GRID_SIZE:
                            map.oGrid[target_x][target_y] = 0
                elif placeTarget:
                    if event.button == 1:
                        if 0 <= target_x < GRID_SIZE and 0 <= target_y < GRID_SIZE:
                            map.oGrid[target_x][target_y] = 2
                    if event.button == 3:
                        if 0 <= target_x < GRID_SIZE and 0 <= target_y < GRID_SIZE:
                            map.oGrid[target_x][target_y] = 0
                elif not placeTarget and not placeSwarm:
                    if event.button == 1: 
                        map.place_obstacle(target_x, target_y, 20, o, length)
                    elif event.button == 3: 
                        if 0 <= target_x < GRID_SIZE and 0 <= target_y < GRID_SIZE:
                            for i in range(target_x - 20, target_x + 20):
                                for j in range(target_y - 20, target_y + 20):
                                    if 0 <= i < map.grid_size and 0 <= j < map.grid_size:
                                        map.oGrid[i][j] = 0
                            map.draw_obstacles_once()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    map.save_to_file('grid.txt')
                if event.key == pygame.K_o:
                    if o == 0:
                        o = 1
                    else:
                        o = 0
                if event.key == pygame.K_t:
                    placeTarget = not placeTarget
                    placeSwarm = False
                if event.key == pygame.K_q:
                    placeSwarm = not placeSwarm
                    placeTarget = False
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    length += 50
                    print("Length increased to:", length)
                elif event.key == pygame.K_MINUS:
                    length -= 50
                    if length < 100:
                        length = 100
                    print("Length decreased to:", length)
                    print(length)
                    
            
    pygame.quit()
    
