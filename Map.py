################################### Imports ########################################
import pygame
import random
import math
import numpy as np

# Simulation Constants
WINDOW_SIZE = 1000
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.DOUBLEBUF)
CLOCK = pygame.time.Clock()
GRID_SIZE = 500
MAXIMUM_DISTANCE = math.sqrt(GRID_SIZE**2 + GRID_SIZE**2)
MAX_ITERATIONS = math.inf
# Colors
WHITE = (255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()

################################### Map Class ########################################
class Map:
    def __init__(self, grid_size, num_obstacles, min_size, max_size):
        self.grid_size = grid_size
        self.oGrid = [[0] * grid_size for _ in range(grid_size)]
        self.obstacles = []
        self.line_width = 1
        self.obstacle_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        
        self.obstacle_surface.fill(WHITE)
        self.load_from_file("grid.txt")
        #self.generate_obstacles(100, 20, 100)
        self.draw_obstacles_once()
    
    
    def draw_obstacles_once(self):      
        # Clear the surface by filling it with a fully transparent color
        self.obstacle_surface.fill((0, 0, 0, 0))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.oGrid[i][j] == 1:
                    color = BLACK
                    pygame.draw.rect(self.obstacle_surface, color, 
                                     (i * (WINDOW_SIZE // self.grid_size), 
                                      j * (WINDOW_SIZE // self.grid_size), 
                                      WINDOW_SIZE // self.grid_size, 
                                      WINDOW_SIZE // self.grid_size))

    def draw(self):
        SCREEN.blit(self.obstacle_surface, (0, 0))

    def generate_obstacles(self, num_obstacles, min_size, max_size):
        for _ in range(num_obstacles):
            size = random.randint(min_size, max_size)
            x = random.randint(0, self.grid_size - size)
            y = random.randint(0, self.grid_size - size)
            # Check for overlap with existing obstacles
            overlap = False
            for obs_x, obs_y, obs_size in self.obstacles:
                if not (x + size < obs_x or x > obs_x + obs_size or
                        y + size < obs_y or y > obs_y + obs_size):
                    overlap = True
                    break
            if not overlap:
                self.obstacles.append((x, y, size))
        # Fill in the grid cells for the obstacles
        for x, y, size in self.obstacles:
            for i in range(x, x + size):
                for j in range(y, y + size):
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        self.oGrid[i][j] = 1
    
    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                for i, line in enumerate(file):
                    # Convert each line back into a list of integers
                    self.oGrid[i] = list(map(int, line.strip().split(' ')))
            self.draw_obstacles_once()  # Redraw the obstacle surface to reflect the loaded grid
            print("File loaded successfully")
        except Exception as e:
            print(f"Error loading from file: {e}")


    def place_obstacle(self, x, y, size):
        for i in range(x - 400, x + 400):
            for j in range(y - 20, y + 20):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    self.oGrid[i][j] = 1
        self.draw_obstacles_once()
    
    def save_to_file(self, filename):
        print(f"Saving to file: {filename}")  # Debug print statement
        with open(filename, 'w') as file:
            for row in self.oGrid:
                file.write(' '.join(map(str, row)) + '\n')
        print("File saved successfully")  # Debug print statement


def BuildMap():
    map = Map(GRID_SIZE, 20, 1, 8)
    
    prevGrid = map.oGrid
    running = True
    iCount = 0
    
    left_mouse_button_down = False
    right_mouse_button_down = False
    while running:        
        iCount += 1
        SCREEN.fill(WHITE)
        map.draw()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        target_x = mouse_x // (WINDOW_SIZE // GRID_SIZE)
        target_y = mouse_y // (WINDOW_SIZE // GRID_SIZE)

        pygame.display.flip()
        #CLOCK.tick(60)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button down
                    left_mouse_button_down = True
                    map.place_obstacle(target_x, target_y, 20)
                elif event.button == 3:  # Right mouse button down
                    right_mouse_button_down = True
                    if 0 <= target_x < GRID_SIZE and 0 <= target_y < GRID_SIZE:
                        for i in range(target_x - 20, target_x + 20):
                            for j in range(target_y - 400, target_y + 400):
                                if 0 <= i < map.grid_size and 0 <= j < map.grid_size:
                                    map.oGrid[i][j] = 0
                        map.draw_obstacles_once()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button up
                    left_mouse_button_down = False
                elif event.button == 3:  # Right mouse button up
                    right_mouse_button_down = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    map.save_to_file('grid.txt')
            
    pygame.quit()
    
#BuildMap()