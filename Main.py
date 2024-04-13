################################### Imports ########################################
import pygame
import random
import math
import numpy as np
from Map import *
# Initialize Pygame
pygame.init()

################################### Global Variables ########################################
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Simulation Constants
WINDOW_SIZE = 1000
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.DOUBLEBUF)
CLOCK = pygame.time.Clock()
MAXIMUM_DISTANCE = math.sqrt(GRID_SIZE**2 + GRID_SIZE**2)
MAX_ITERATIONS = math.inf

# PSO Constants
C1 = 0.1
C2 = 0.1
INERTIA_WEIGHT = 0.9

# Swarm Constants
MAX_VELOCITY = 10
DISPERSE_RADIUS = 4
NUMBER_OF_PARTICLES = 100
MAX_DISCOVERY = 100
MAX_PHEROMONES = 7000
MEMGRID_SIZE = 0.1


STATE_SEEK = 0
STATE_ATTACK = 1

################################### Target Class ########################################

class Target:
    def __init__(self, x, y):
        self.position = [x, y]
    
    def isDiscoverable(self, px, py):
        distance = math.sqrt((px - self.position[0])**2 + (py - self.position[1])**2)
        return distance < GRID_SIZE / 4
    
    def getDistance(self, px, py):
        return math.sqrt((px - self.position[0])**2 + (py - self.position[1])**2)
        
    def isAttacked(self, px, py):
        distance = math.sqrt((px - self.position[0])**2 + (py - self.position[1])**2)
        return distance < 1
    
    def draw(self):
        pygame.draw.circle(SCREEN, (255, 0, 0), (self.position[0] * (WINDOW_SIZE // GRID_SIZE) \
            + (WINDOW_SIZE // GRID_SIZE // 2), self.position[1] * (WINDOW_SIZE // GRID_SIZE) \
                + (WINDOW_SIZE // GRID_SIZE // 2)), 10)

################################### Swarm Memory Class ########################################
class SwarmMemory:
    def __init__(self, nParticles, c1, c2, inertiaWeight):
        self.nParticles = nParticles
        self.c1 = c1
        self.c2 = c2
        self.inertiaWeight = inertiaWeight
        self.gBestPos = (0, 0)
        self.gBestFitness = 0
        self.i = 0
        self.memGrid = [[0] * int(GRID_SIZE * MEMGRID_SIZE) for _ in range(int(GRID_SIZE * MEMGRID_SIZE))]
        self.update = False
        self.nParticles = None
        self.pheromones = []
        self.mode = -1
        self.prevPos = []
        
    def placePheromone(self, pos):
        self.pheromones.append(pos)
    
    def vaporise(self):
        if len(self.pheromones) > MAX_PHEROMONES:
            self.pheromones.pop(0)
            
    
    def calculateAvrPosition(self):
        # Calculate the average position of all particles
        avx = 0
        avy = 0
        for particle in self.particles:
            avx += particle.position[0]
            avy += particle.position[1]
        avx /= len(self.particles)
        avy /= len(self.particles)
        return (avx, avy)

    def setParticles(self, particles):
        self.particles = particles

        
    def restartOptimisation(self):
        self.gBestFitness = 0
        for particle in self.particles:
            particle.pBestFitness = 0
        self.update = False
    
    def recordPosition(self, pos):
        self.prevPos.append(pos)
        
################################### Particle Class ########################################
class Agent:
    def __init__(self, x, y, id, memory):
        self.id = id
        self.position = [x, y]
        self.memory = memory
        self.velocity = [random.uniform(-MAX_VELOCITY, MAX_VELOCITY), random.uniform(-MAX_VELOCITY, MAX_VELOCITY)]
        self.pBestPosition = [x, y]
        self.pBestFitness = 0
        self.state = STATE_SEEK
        self.memory.placePheromone(self.position)

    def activate(self, grid, targets):
        # Check if the agent is in the vicinity of a target
        self.state = STATE_SEEK
        for target in targets:
            if target.isDiscoverable(self.position[0], self.position[1]):
                self.state = STATE_ATTACK
                break

        
        # Disperse from other particles
        self.disperse(self.memory.particles, grid)
        
        # Update velocity and position
        self.update_velocity(self.memory.gBestPos, self.memory.inertiaWeight, self.memory.c1, self.memory.c2)
        
        # Update position and fitness
        self.update_position(grid)
        
        # Calculate fitness of current position
        fitness = self.calculateFitness(targets)
        
        # If the fitness of the current position is better than the personal best, update the personal best
        if fitness > self.pBestFitness:
            self.pBestPosition = self.position
            self.pBestFitness = fitness
            # If the personal best is better than the global best, update the global best
            if self.pBestFitness > self.memory.gBestFitness:
                self.memory.gBestPos = self.pBestPosition
                self.memory.gBestFitness = self.pBestFitness
        
        # Map the agent's position to the memory grid
        gridX = int(round(self.position[0] * MEMGRID_SIZE))
        gridY = int(round(self.position[1] * MEMGRID_SIZE))     
        if gridX >= len(self.memory.memGrid):
            gridX = len(self.memory.memGrid) - 1
        if gridY >= len(self.memory.memGrid[0]):
            gridY = len(self.memory.memGrid[0]) - 1
            
        # Increment the memory grid value if the agent is in the seek state
        if self.state == STATE_SEEK and \
            self.memory.memGrid[gridX][gridY] < MAX_DISCOVERY \
                and self.memory.i % 2 == 0 and self.state == STATE_SEEK: 
            self.memory.memGrid[gridX][gridY] += 1
            self.memory.placePheromone(self.position)
            self.memory.update = True
                
        # If the agent is in the attack state check if it has reached a target
        elif self.state == STATE_ATTACK:
            for target in targets:
                if target.isAttacked(self.position[0], self.position[1]):
                    targets.remove(target)
                    self.state = STATE_SEEK
                    self.memory.restartOptimisation()
        
        # Restart the optimisation if the last particle has updated the memory grid
        if self.id == NUMBER_OF_PARTICLES - 1:
            self.memory.i += 1
            if self.state == STATE_SEEK and self.memory.i % 4 == 0 and self.memory.update == True:
                self.memory.restartOptimisation()
        self.memory.vaporise()

        self.draw()

    
    def disperse(self, particles, grid):
        # Calculate the average direction to move away from other particles
        for p in particles:
            if p != self:
                distance = math.sqrt((p.position[0] - self.position[0])**2 + (p.position[1] - self.position[1])**2)
                if distance < DISPERSE_RADIUS and distance > 0:  
                    dx = self.position[0] - p.position[0]
                    dy = self.position[1] - p.position[1]
                    self.velocity[0] += dx / distance  
                    self.velocity[1] += dy / distance
    
    def calculateFitness(self, targets):
        distance = MAXIMUM_DISTANCE
                    
        if self.state == STATE_ATTACK:
            for target in targets:
                if target.isDiscoverable(self.position[0], self.position[1]):
                    newDistance = target.getDistance(self.position[0], self.position[1])
                    if newDistance < distance:
                        distance = newDistance
            fitness = 2 - (distance / MAXIMUM_DISTANCE)
        else:
            if self.memory.mode == 0:
                gridX = int(round(self.position[0] / 10))
                gridY = int(round(self.position[1] / 10))     
                # Clamping grid_x and grid_y to the maximum index of memGrid
                if gridX >= len(self.memory.memGrid):
                    gridX = len(self.memory.memGrid) - 1
                if gridY >= len(self.memory.memGrid[0]):
                    gridY = len(self.memory.memGrid[0]) - 1

                fitness = self.memory.memGrid[gridX][gridY]
                fitness = 1 + (fitness / MAX_DISCOVERY)
                fitness = 2 - fitness
            elif self.memory.mode == 1:
                avx, avy = self.memory.calculateAvrPosition()
                distance = math.sqrt((self.position[0] - avx)**2 + (self.position[1] - avy)**2)
                fitness = (distance / MAXIMUM_DISTANCE) 
            else:
                fitness = random.uniform(0, 1)
        return fitness

    def update_velocity(self, gBestPos, inertiaWeight, c1, c2):
        r1 = random.random()
        r2 = random.random()
        self.velocity[0] = inertiaWeight * self.velocity[0] + c1 * r1 * (self.pBestPosition[0] - self.position[0]) + c2 * r2 * (gBestPos[0] - self.position[0])
        self.velocity[1] = inertiaWeight * self.velocity[1] + c1 * r1 * (self.pBestPosition[1] - self.position[1]) + c2 * r2 * (gBestPos[1] - self.position[1])
        
        # Apply velocity clamping
        velocity_mag = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        while velocity_mag > MAX_VELOCITY or velocity_mag < -MAX_VELOCITY:
            # Scale down velocity to the maximum allowed
            self.velocity[0] /= 2
            self.velocity[1] /= 2
            velocity_mag = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            #self.velocity[0] = (self.velocity[0] / velocity_mag) * MAX_VELOCITY 
            #self.velocity[1] = (self.velocity[1] / velocity_mag) * MAX_VELOCITY

    def update_position(self, grid):
        # Proposed new position based on velocity
        proposed_x = int(round(self.position[0] + self.velocity[0]))
        proposed_y = int(round(self.position[1] + self.velocity[1]))

        # Ensure proposed_x and proposed_y are within grid bounds
        proposed_x = max(0, min(GRID_SIZE - 1, proposed_x))
        proposed_y = max(0, min(GRID_SIZE - 1, proposed_y))

        # Check direct path
        if grid[proposed_x][proposed_y] != 1:
            # Reflect off boundaries
            if proposed_x <= 0 or proposed_x >= GRID_SIZE - 1:
                self.velocity[0] *= -1
            if proposed_y <= 0 or proposed_y >= GRID_SIZE - 1:
                self.velocity[1] *= -1

            # Update position with adjusted velocity
            self.position[0] += self.velocity[0]
            self.position[1] += self.velocity[1]
            self.position = [proposed_x, proposed_y]
        else:
            # Attempt to "slide" along the obstacle by checking alternative movements
            alternative_x = int(self.position[0] + self.velocity[0])
            alternative_y = int(self.position[1] + self.velocity[1])
            
            # Ensure alternative_x and alternative_y are within grid bounds
            alternative_x = max(0, min(GRID_SIZE - 1, alternative_x))
            alternative_y = max(0, min(GRID_SIZE - 1, alternative_y))
            
            if grid[alternative_x][int(self.position[1])] != 1:
                self.position[0] = alternative_x  # Move horizontally only
            if grid[int(self.position[0])][alternative_y] != 1:
                self.position[1] = alternative_y  # Move vertically only
        
        if self.id == NUMBER_OF_PARTICLES - 1: 
            self.memory.recordPosition(self.position)

    
    def draw(self):
        radius = (WINDOW_SIZE // GRID_SIZE) # Ensure minimum radius of 1
        center_x = self.position[0] * (WINDOW_SIZE // GRID_SIZE) + (WINDOW_SIZE // GRID_SIZE // 2)
        center_y = self.position[1] * (WINDOW_SIZE // GRID_SIZE) + (WINDOW_SIZE // GRID_SIZE // 2)
        pygame.draw.circle(SCREEN, BLUE, (center_x, center_y), radius)



################################### Simulation ########################################

def draw_memory_visualization(memory, window_size, grid_size, visualiseMemGrid, visualisePrevPos, visualisePath):
    cell_size = window_size / grid_size 
    max_visits = MAX_DISCOVERY
    if visualiseMemGrid:
        for x in range(len(memory.memGrid)):
            for y in range(len(memory.memGrid[x])):
                visits = memory.memGrid[x][y]
                # Calculate the intensity of fade based on visits, ranging from 0 (no visits) to 1 (max visits)
                fade_intensity = visits / max_visits if max_visits > 0 else 0
                # Interpolate between white and red; reduce green and blue components to fade to red
                green_blue_intensity = int(255 * (1 - fade_intensity))
                color = (255, green_blue_intensity, green_blue_intensity)  # Red stays at 255, green and blue fade to 0
                # Draw the cell with the calculated color
                if memory.memGrid[x][y] > 0:
                    pygame.draw.rect(SCREEN, color, (x * cell_size / MEMGRID_SIZE, y * cell_size / MEMGRID_SIZE, cell_size / MEMGRID_SIZE, cell_size / MEMGRID_SIZE))
    
    if visualisePrevPos:
        for (x, y) in memory.pheromones:
            pygame.draw.rect(SCREEN, (0, 255, 0), (x * cell_size, y * cell_size, cell_size, cell_size))
   
    if visualisePath:
        for (x, y) in memory.prevPos:
            pygame.draw.rect(SCREEN, (0, 0, 0), (x * cell_size, y * cell_size, cell_size, cell_size))

            
def runSimulation():
    # Initialize grid and obstacles
    visualiseMemGrid = False
    visualisePrevPos = False
    visaulisePath = False
    
    targets = [Target(0, 0)]
    map = Map(GRID_SIZE, 20, 1, 8)
    memory = SwarmMemory(NUMBER_OF_PARTICLES, C1, C2, INERTIA_WEIGHT)
    particles = []
    for i in range(NUMBER_OF_PARTICLES):
        particles.append(Agent(500, 500, i, memory))

    memory.setParticles(particles)
    running = True
    iCount = 0
    while running:
        iCount += 1
        
        if memory.mode == 0:
            visualiseMemGrid = True
            visualisePrevPos = False
        elif memory.mode == 1:
            visualisePrevPos = True
            visualiseMemGrid = False
        else:
            visualiseMemGrid = False
            visualisePrevPos = False
        
        #if targets == []:
        #    running = False
        #    print ("Simulation ended after ", iCount, " iterations.")
        SCREEN.fill(WHITE)
        draw_memory_visualization(memory, WINDOW_SIZE, GRID_SIZE, visualiseMemGrid, visualisePrevPos, visaulisePath)
        map.draw()
        
        mouse_x, mouse_y = pygame.mouse.get_pos()
        target_x = mouse_x // (WINDOW_SIZE // GRID_SIZE)
        target_y = mouse_y // (WINDOW_SIZE // GRID_SIZE)

        
        iCount += 1
        
        for particle in particles:
            particle.activate(map.oGrid, targets)
        

        for target in targets:
            target.draw()
        
        pygame.display.flip()
        CLOCK.tick(60)
        #print(CLOCK.get_fps())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                target_x = mouse_x // (WINDOW_SIZE // GRID_SIZE)
                target_y = mouse_y // (WINDOW_SIZE // GRID_SIZE)
                t = Target(target_x, target_y)
                targets.append(t)
                memory.restartOptimisation()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    memory.mode = 0
                if event.key == pygame.K_1:
                    memory.mode = 1
                if event.key == pygame.K_p:
                    visaulisePath = not visaulisePath
                    draw_memory_visualization(memory, WINDOW_SIZE, GRID_SIZE, visualiseMemGrid, visualisePrevPos, visaulisePath)
    #pygame.quit()
    

def Main():    
    # Prompt the user to enter the 'run' command to start the simulation
    start_command = input("")
    
    if start_command.lower() == 'run':
        runSimulation()
    elif start_command.lower() == 'build':
        BuildMap()
    else:
        print("Invalid command. Please enter 'run' to start the simulation or 'build' to build the map.")
        Main()
runSimulation()