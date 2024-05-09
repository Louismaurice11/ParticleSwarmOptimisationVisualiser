################################### Imports ########################################
import pygame
import random
import math
from Map import *
import csv
import numpy as np
import scipy.stats as stats
import time
import pandas as pd

################################### Global Variables ########################################
# Simulation Constants
WINDOW_SIZE = 1000
GRID_SIZE = 500
SCREEN = 0
CLOCK = pygame.time.Clock()
MAXIMUM_DISTANCE = math.sqrt(GRID_SIZE**2 + GRID_SIZE**2)
MAX_ITERATIONS = 15000
MEMGRID_SIZE = 0.1

# Control variables
INERTIA_WEIGHT = 0.8 # Inertia weight
REOPTIMISATION_RATE = 5 # Reset the optimisation process every n iterations
DISPERSE_RADIUS = 10 # Radius of dispersion from other particles
AVOIDANCE_RADIUS = 5 # Radius of avoidance from obstacles
DEACTIVATION_RADIUS = 75 # Radius of deactivation from the leader
MAX_VELOCITY = 10 # Maximum velocity of the particles
AGENT_VISION = 50 # Vision range of the particles
MAX_DISCOVERY = 0 # Maximum discovery value
MAX_PHEROMONES = 0 # Maximum number of pheromones
C1 = 0.5 # Cognitive coefficient
C2 = 0.5 # Social coefficient

# Particle States
STATE_SEEK = 0
STATE_ATTACK = 1
STATE_DEACTIVATE = 2
FITNESS_MODE_DISCOVERY = 0
FITNESS_MODE_PHEROMONE = 1
FITNESS_MODE_HYBRID = 2





################################### Target Class ########################################

class Target:
    
    # Initialize the target
    def __init__(self, x, y):
        self.position = [x, y]
    
    # Check if the target is discoverable
    def isDiscoverable(self, px, py):
        distance = math.sqrt((px - self.position[0])**2 + (py - self.position[1])**2)
        return distance < AGENT_VISION
    
    # Get the distance between the target and a specified position
    def getDistance(self, px, py):
        return math.sqrt((px - self.position[0])**2 + (py - self.position[1])**2)
    
    # Check if the target is attacked
    def isAttacked(self, px, py):
        distance = math.sqrt((px - self.position[0])**2 + (py - self.position[1])**2)
        return distance < 5
    
    # Draw the target
    def draw(self):
        pygame.draw.circle(SCREEN, (255, 0, 0), (self.position[0] * (WINDOW_SIZE // GRID_SIZE) \
            + (WINDOW_SIZE // GRID_SIZE // 2), self.position[1] * (WINDOW_SIZE // GRID_SIZE) \
                + (WINDOW_SIZE // GRID_SIZE // 2)), 10)





################################### Swarm Memory Class ########################################

class SwarmMemory:
    # Initialize the swarm memory
    def __init__(self, nParticles, c1, c2, inertiaWeight, mode):
        # Swarm memory
        self.gBestPos = (0, 0) # Global best position
        self.gBestFitness = 0 # Global best fitness
        self.iCount = 0 # Iteration count
        self.memGrid = [[0] * int(GRID_SIZE * MEMGRID_SIZE) for _ in range(int(GRID_SIZE * MEMGRID_SIZE))] # Memory grid
        self.pheromones = [] # List of pheromones placed by the particles
        self.nPheromones = 0 # Number of pheromones placed
        
        # Swarm properties
        self.mode = mode # 0 = Discovery, 1 = Pheromone, 2 = Both
        self.nParticles = nParticles # Number of particles in the swarm
        self.c1 = c1 # Cognitive coefficient
        self.c2 = c2 # Social coefficient
        self.inertiaWeight = inertiaWeight # Inertia weight
        self.avrPheromonePos = (0, 0) # Average position of all pheromones
        self.avrPos = (0,0) # Average position of all particles
        self.leader = 0 # Index of the leader particle
        self.particles = [] # List of particles in the swarm
        
        # Visualisation properties
        self.prevPos = [] # List of previous positions of the leader
        
        #Other
        self.restartOptimisationFlag = False # Flag to indicate if the memory grid has been updated
        
    # Calculate the average position of all particles
    def calculateAvrPos(self):
        x = 0
        y = 0
        i = 0
        for particle in self.particles:
            if particle.state == STATE_DEACTIVATE:
                continue
            x += particle.position[0]
            y += particle.position[1]
            i += 1
        x /= i
        y /= i
        self.avrPos = (x, y)
    
    def incrementMemory(self, x, y):
        # Map the agent's position to the memory grid
        mappedX = int(round(x * MEMGRID_SIZE))
        mappedY = int(round(y * MEMGRID_SIZE))     
        if mappedX >= len(self.memGrid):
            mappedX = len(self.memGrid) - 1
        if mappedY >= len(self.memGrid[0]):
            mappedY = len(self.memGrid[0]) - 1
            
        if self.memGrid[mappedX][mappedY] < MAX_DISCOVERY:
            self.memGrid[mappedX][mappedY] += 1
        
    
    # Place a pheromone at the specified position
    def placePheromone(self, pos):
        self.pheromones.append(pos)
        self.nPheromones += 1
    
    
    # Vaporise the oldest pheromone
    def vaporise(self):
        if len(self.pheromones) > MAX_PHEROMONES:
            self.pheromones.pop(0)
            self.nPheromones -= 1
    
    
    # Calculate the average position of all pheromones
    def calculateAvrPheromone(self):
        x = 0
        y = 0
        i = 0
        for pos in self.pheromones:
            x += pos[0]
            y += pos[1]
            i += 1
        x /= i
        y /= i
        self.avrPheromonePos = (x, y)
        
        
    # Set the particles in the swarm
    def setParticles(self, particles):
        self.particles = particles


    # Restart the optimisation process
    def restartOptimisation(self):
        self.gBestFitness = 0
        for particle in self.particles:
            particle.pBestFitness = 0
        self.restartOptimisationFlag = False
    
    
    # Record the position of a particle
    def recordPosition(self, pos):
        self.prevPos.append(pos)
    
    
    # Reset the memory grid and previous positions
    def resetPrevMem(self):
        self.prevPos = []
        self.memGrid = [[0] * int(GRID_SIZE * MEMGRID_SIZE) for _ in range(int(GRID_SIZE * MEMGRID_SIZE))]
        self.pheromones = []
        self.placePheromone(self.particles[self.leader].position)
        self.avrPos = []
        
        
        
        
        
################################### Particle Class ########################################
class Agent:
    
    # Initialize the particle
    def __init__(self, x, y, id, memory):
        # Agent Memory
        self.pBestPosition = [x, y] # Personal best position
        self.pBestFitness = 0 # Personal best fitness
        
        # Agent Properties
        self.memory = memory # Shared swarm memory
        self.id = id # Particle ID
        self.state = STATE_SEEK # Particle state
        self.position = [x, y] # Particle grid position
        self.velocity = [random.uniform(-MAX_VELOCITY, MAX_VELOCITY), random.uniform(-MAX_VELOCITY, MAX_VELOCITY)] # Particle velocity

        # Place a pheromone at the particle's initial position
        self.memory.placePheromone(self.position) 
        
    # Activate the particle
    def activate(self, grid, targets):
        # If the particle is the leader
        if self.id == self.memory.leader:
            # Update the iteration count
            self.memory.iCount += 1
            # If the particle is in the seek state and reoptimisation flag is true, restart the optimisation process
            if self.state == STATE_SEEK and \
                    self.memory.iCount % REOPTIMISATION_RATE == 0:
                self.memory.restartOptimisationFlag = True
            
            # Calculate the average position of all particles
            self.memory.calculateAvrPos()
            # Record the position of the leader
            self.memory.recordPosition(self.position)
            
            # If the particle is in the discovery mode, calculate the average pheromone position
            if self.memory.mode == 1 or self.memory.mode == 2:
                self.memory.calculateAvrPheromone()
        
        # Choose the particle's next state
        self.chooseState(targets)
        
        if self.state != STATE_DEACTIVATE:
            # Disperse from other particles
            self.disperse(self.memory.particles)
            
            # Avoid obstacles
            self.avoidObstacles(grid)
            
            # Update velocity and position
            self.updateVelocity(self.memory.gBestPos, self.memory.inertiaWeight, self.memory.c1, self.memory.c2) 
            
            # Update position and fitness
            self.updatePosition(grid)
            
            
            # Calculate fitness of current position 
            fitness = self.calculateFitness(targets, grid)
            
            # If the fitness of the current position is better than the personal best, update the personal best
            if fitness > self.pBestFitness:
                self.pBestPosition = self.position
                self.pBestFitness = fitness
            # If the personal best is better than the global best, update the global best
            if fitness > self.memory.gBestFitness:
                self.memory.gBestPos = self.position
                self.memory.gBestFitness = fitness
            
            # Update the memory grid    
            if self.memory.mode != FITNESS_MODE_PHEROMONE:
                self.memory.incrementMemory(self.position[0], self.position[1])
            if self.memory.mode != FITNESS_MODE_DISCOVERY:
                self.memory.placePheromone(self.position)
            
            # If the agent is in the attack state check if it has reached a target
            if self.state == STATE_ATTACK:
                for target in targets:
                    if target.isAttacked(self.position[0], self.position[1]):
                        targets.remove(target)
                        self.memory.restartOptimisationFlag = True
            
            # Restart the optimisation if the last particle has updated the memory grid or placed a pheromone
            if self.id == self.memory.leader and self.state == STATE_SEEK and self.memory.restartOptimisationFlag == True:
                    self.memory.restartOptimisation()
            self.memory.vaporise()
        
        # Draw the particle
        self.draw()

    # Choose agent's next state
    def chooseState(self, targets):
        # Check if the particle is lost
        if self.checkLost():
            self.state = STATE_DEACTIVATE
            if self.id == self.memory.leader:
                self.memory.leader += 1
            return
        
        # Check if the particle is close to a target
        for target in targets:
            if target.isDiscoverable(self.position[0], self.position[1]):
                # If the particle is close to a target, switch to attack state
                self.state = STATE_ATTACK
                return
        
        # Else, the particle is in the seek state
        self.state = STATE_SEEK 

        
    # Check if the particle is lost
    def checkLost(self):
        self.memory.calculateAvrPos()
        distance = math.sqrt((self.position[0] - self.memory.avrPos[0])**2 + (self.position[1] - self.memory.avrPos[1])**2)
        if distance > DEACTIVATION_RADIUS:
            return True
        else:
            return False
    


    # Disperse the particle from other particles
    def disperse(self, particles):
        # for each particle in the swarm
        for p in particles:
            # if the particle is not the current particle
            if p != self:
                # calculate the distance between the current particle and the other particle
                distance = math.sqrt((p.position[0] - self.position[0])**2 + (p.position[1] - self.position[1])**2)
                
                # if the distance is less than the disperse radius and greater than 0
                if distance < DISPERSE_RADIUS and distance > 0:  
                    # Update the velocity of the current particle to disperse from the other particle                    
                    dx = self.position[0] - p.position[0]
                    dy = self.position[1] - p.position[1]
                    self.velocity[0] += (dx / distance) 
                    self.velocity[1] += (dy / distance)
        
    # Avoid obstacles
    def avoidObstacles(self, grid):
        # for each cell in the avoidance radius
        for x in range(-AVOIDANCE_RADIUS, AVOIDANCE_RADIUS+1):
            for y in range(-AVOIDANCE_RADIUS, AVOIDANCE_RADIUS+1):
                # if the cell is out of bounds, skip
                if self.position[0] + x < 0 or self.position[0] + x >= GRID_SIZE or self.position[1] + y < 0 or self.position[1] + y >= GRID_SIZE:
                    continue
                # if the cell is an obstacle, update the velocity to avoid the obstacle
                elif grid[self.position[0] + x][self.position[1] + y] == 1:
                    dx = self.position[0] - (self.position[0] + x)
                    dy = self.position[1] - (self.position[1] + y)
                    distance = math.sqrt(dx**2 + dy**2)
                    self.velocity[0] += (dx / distance) 
                    self.velocity[1] += (dy / distance) 
                
    # Calculate the fitness of the current position
    def calculateFitness(self, targets, grid):
        # Set the fitness to 0
        fitness = 0
        # If the particle is in the attack state, calculate the fitness based on the distance to the target        
        if self.state == STATE_ATTACK:
            distance = MAXIMUM_DISTANCE
            for target in targets:
                if target.isDiscoverable(self.position[0], self.position[1]):
                    newDistance = target.getDistance(self.position[0], self.position[1])
                    if newDistance < distance:
                        distance = newDistance
            fitness = 2 - (distance / MAXIMUM_DISTANCE)
            
        # If the particle is in the seek state, calculate the fitness based on the discovery and / or pheromone values
        elif self.state == STATE_SEEK:
            # Map the agent's position to the memory grid
            gridX = int(round(self.position[0] / 10))
            gridY = int(round(self.position[1] / 10))     
            
            # Ensure gridX and gridY are within the bounds of the memory grid to prevent index out of range errors          
            if gridX >= len(self.memory.memGrid):
                gridX = len(self.memory.memGrid) - 1
            if gridY >= len(self.memory.memGrid[0]):
                gridY = len(self.memory.memGrid[0]) - 1
            
            # Calculate the discovery amount
            discovery = self.memory.memGrid[gridX][gridY]
            # Normalise the discovery value
            discovery = discovery / MAX_DISCOVERY
            if discovery >= 1:
                dFitness = random.uniform(0, 0.01)
            else:
                dFitness = 1 - discovery
            
            # Calculate the pheromone amount
            distance = math.sqrt((self.position[0] - self.memory.avrPheromonePos[0])**2 + (self.position[1] - self.memory.avrPheromonePos[1])**2)
            # Normalise the distance value
            pFitness = (distance / MAXIMUM_DISTANCE)
            
            
            # Calculate the fitness based on the mode
            if self.memory.mode == FITNESS_MODE_DISCOVERY:
                fitness = dFitness
            elif self.memory.mode == FITNESS_MODE_PHEROMONE:
                fitness = pFitness
            elif self.memory.mode == FITNESS_MODE_HYBRID:
                fitness = (dFitness + pFitness) / 2
            else:
                fitness = random.uniform(0, 1)
                
            
            #fitness = self.obstacleAwareness(grid, fitness)
            
        # Return the fitness
        return fitness

    # Avoid obstacles - Unused due to performance issues - recomendation for future work
    def obstacleAwareness(self, grid, fitness):
        # for each cell in the avoidance radius
        for x in range(-AVOIDANCE_RADIUS, AVOIDANCE_RADIUS+1):
            for y in range(-AVOIDANCE_RADIUS, AVOIDANCE_RADIUS+1):
                # if the cell is out of bounds, skip
                if self.position[0] + x < 0 or self.position[0] + x >= GRID_SIZE or self.position[1] + y < 0 or self.position[1] + y >= GRID_SIZE:
                    continue
                # if the cell is an obstacle, update the velocity to avoid the obstacle
                elif grid[self.position[0] + x][self.position[1] + y] == 1:
                    fitness -= 0.1
                    
        return fitness
        
    # Update the velocity of the particle
    def updateVelocity(self, gBestPos, inertiaWeight, c1, c2):
        # Calculate random values
        r1 = random.random()
        r2 = random.random()
        
        # Update the velocity using the PSO formula
        self.velocity[0] = inertiaWeight * self.velocity[0] + c1 * r1 * (self.pBestPosition[0] - self.position[0]) + c2 * r2 * (gBestPos[0] - self.position[0])
        self.velocity[1] = inertiaWeight * self.velocity[1] + c1 * r1 * (self.pBestPosition[1] - self.position[1]) + c2 * r2 * (gBestPos[1] - self.position[1])
        
        # Apply velocity clamping
        vMagnitude = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        while vMagnitude > MAX_VELOCITY or vMagnitude < -MAX_VELOCITY:
            # Scale down velocity to the maximum allowed
            vMagnitude = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            self.velocity[0] = (self.velocity[0] / vMagnitude) * MAX_VELOCITY 
            self.velocity[1] = (self.velocity[1] / vMagnitude) * MAX_VELOCITY


    # Update the position of the particle
    def updatePosition(self, grid):
        oPos = self.position
        # Proposed new position based on velocity
        proposedX = int(round(self.position[0] + self.velocity[0]))
        proposedY = int(round(self.position[1] + self.velocity[1]))

        # Ensure proposed_x and proposed_y are within grid bounds
        proposedX = max(0, min(GRID_SIZE - 1, proposedX))
        proposedY = max(0, min(GRID_SIZE - 1, proposedY))

        # Check if the proposed position is an obstacle
        if grid[proposedX][proposedY] != 1:
            # Update position 
            self.position = [proposedX, proposedY]
        else:
            if grid[proposedX][int(self.position[1])] != 1:
                self.position[0] = proposedX  # Move horizontally only
            if grid[int(self.position[0])][proposedY] != 1:
                self.position[1] = proposedY  # Move vertically only
                

    # Draw the particle
    def draw(self):
        if self.id == self.memory.leader:
            color = (255, 0, 0) # Red - Leader
        else:
            color = (0, 0, 0) # Black - Standard Particle
        radius = (WINDOW_SIZE // GRID_SIZE) 
        center_x = self.position[0] * (WINDOW_SIZE // GRID_SIZE) + (WINDOW_SIZE // GRID_SIZE // 2)
        center_y = self.position[1] * (WINDOW_SIZE // GRID_SIZE) + (WINDOW_SIZE // GRID_SIZE // 2)
        pygame.draw.circle(SCREEN, color, (center_x, center_y), radius)





################################### Simulation ########################################

# Draw the memory grid, previous positions and path
def drawMemoryVisualisation(memory, window_size, grid_size, visualiseMemGrid, visualisePrevPos, visualisePath):
    cell_size = window_size / grid_size 
    max_visits = MAX_DISCOVERY
    if visualiseMemGrid:
        for x in range(len(memory.memGrid)):
            for y in range(len(memory.memGrid[x])):
                visits = memory.memGrid[x][y]
                fade_intensity = visits / max_visits if max_visits > 0 else 0
                green_blue_intensity = int(255 * (1 - fade_intensity))
                color = (255, green_blue_intensity, green_blue_intensity)  
                if memory.memGrid[x][y] > 0:
                    pygame.draw.rect(SCREEN, color, (x * cell_size / MEMGRID_SIZE, y * cell_size / MEMGRID_SIZE, cell_size / MEMGRID_SIZE, cell_size / MEMGRID_SIZE))
    if visualisePrevPos:
        for (x, y) in memory.pheromones:
            pygame.draw.rect(SCREEN, (0, 255, 0), (x * cell_size, y * cell_size, cell_size, cell_size))
        pygame.draw.rect(SCREEN, (0, 0, 255), (memory.avrPheromonePos[0] * cell_size, memory.avrPheromonePos[1] * cell_size, cell_size*10, cell_size*10))


# Draw the path of the leader particle
def drawPath(memory, window_size, grid_size):
    cell_size = window_size / grid_size 
    for i in range(len(memory.prevPos)):
                if i == 0:
                    continue
                pygame.draw.line(SCREEN, (0, 0, 0), (memory.prevPos[i][0] * cell_size, memory.prevPos[i][1] * cell_size), (memory.prevPos[i - 1][0] * cell_size, memory.prevPos[i - 1][1] * cell_size), 1)


# Run the simulation
def runSimulation(mode, testMap, swarmSize):
    global MAX_DISCOVERY
    global MAX_PHEROMONES
    MAX_DISCOVERY = swarmSize * 10
    MAX_PHEROMONES = swarmSize * 10
    
    visaliseMemGrid = False
    visualisePrevPos = False
    visaulisePath = False
    visualiseMemory = False
    map = Map(GRID_SIZE, testMap)
    spawn = map.loadSpawnPos()
    memory = SwarmMemory(swarmSize, C1, C2, INERTIA_WEIGHT, mode)
    particles = []
    
    for i in range(swarmSize):
        particles.append(Agent(spawn[0], spawn[1], i, memory))

    memory.setParticles(particles)
    running = True
    iCount = 0
    targets = []
    targetPositions = map.loadTargetPositions()
    for pos in targetPositions:
        t = Target(pos[0], pos[1])
        targets.append(t)
        
    while running and iCount < MAX_ITERATIONS:
        if memory.mode == FITNESS_MODE_DISCOVERY:
            visualiseMemGrid = True
            visualisePrevPos = False
        elif memory.mode == FITNESS_MODE_PHEROMONE:
            visualisePrevPos = True
            visualiseMemGrid = False
        elif memory.mode == FITNESS_MODE_HYBRID:
            visualiseMemGrid = True
            visualisePrevPos = True
        else:
            visualiseMemGrid = False
            visualisePrevPos = False
        
        if targets == []:
            running = False
            print ("Simulation ended after ", iCount, " iterations.")
        
        
        SCREEN.fill((255, 255, 255))
        if visualiseMemory:
            drawMemoryVisualisation(memory, WINDOW_SIZE, GRID_SIZE, visualiseMemGrid, visualisePrevPos, visaulisePath)
        if visaulisePath:
            drawPath(memory, WINDOW_SIZE, GRID_SIZE)
        map.draw()
        
        mouse_x, mouse_y = pygame.mouse.get_pos()
        target_x = mouse_x // (WINDOW_SIZE // GRID_SIZE)
        target_y = mouse_y // (WINDOW_SIZE // GRID_SIZE)
        
        for particle in particles:
            particle.activate(map.oGrid, targets)
        
        iCount += 1

        for target in targets:
            target.draw()
        
        pygame.display.flip()
        #CLOCK.tick(60)
        #print(CLOCK.get_fps())
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                print ("Closing simulation.")
                return -1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    memory.resetPrevMem()
                    memory.mode = FITNESS_MODE_DISCOVERY
                if event.key == pygame.K_1:
                    memory.resetPrevMem()
                    memory.mode = FITNESS_MODE_PHEROMONE
                if event.key == pygame.K_2:
                    memory.resetPrevMem()
                    memory.mode = FITNESS_MODE_HYBRID
                if event.key == pygame.K_p:
                    visaulisePath = not visaulisePath
                    drawMemoryVisualisation(memory, WINDOW_SIZE, GRID_SIZE, visualiseMemGrid, visualisePrevPos, visaulisePath)
                if event.key == pygame.K_v:
                    visualiseMemory = not visualiseMemory
    return iCount

# Run the experiment
def runExperiment():
    # Clear the contents of the results file
    open('mean_results.csv', 'w').close()
    
    # Initialize the results list
    results = [] 
    mean_results = []
    eCount = 0
    repititions = 10
    totalExperiments = 3 * 3 * 3 * repititions
    
    # Run the simulation for each memory mode, coefficient ratio and test map
    for memoryMode in range(1,3):
        for testMap in range(1, 4):
            for i in range(0,3):
                timeTaken = []
                for j in range(repititions):
                    eCount += 1
                    print(f"Running experiment {eCount} out of {totalExperiments}")
                    if i == 0:
                        swarmSize = 50
                    elif i == 1:
                        swarmSize = 100
                    elif i == 2:
                        swarmSize = 150
                    # Run the simulation
                    start_time = time.time()
                    result = runSimulation(memoryMode, testMap, swarmSize)
                    if result == -1:
                        return
                    end_time = time.time()
                    # Calculate the time taken
                    time_taken = end_time - start_time
                    # Add the results to the results list
                    results.append([time_taken, memoryMode, swarmSize, testMap])
                    
                    # Add the time taken to the time taken list
                    timeTaken.append(time_taken)
                    
                # Calculate the mean, standard deviation and standard error of the mean
                mean = round(np.mean(timeTaken), 3) # Used to show the average time taken
                stdDev = round(np.std(timeTaken), 3) # Used to show the consistency of the results
                sem = round(stats.sem(timeTaken), 3) # Used to show the reliability of the mean
                
                # Add the mean values to the mean results list
                mean_results.append([mean, stdDev, sem, memoryMode, swarmSize, testMap])
                
    # Save the results to a file
    print(f"Saving to file: {'results.csv'}")  
    with open("results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time Taken", "Memory Mode", "Swarm Size", "Map"])  # write header
        for result in results:
            writer.writerow(result)
    print("File saved successfully")  
    print(f"Saving mean values to file: {'mean_results.csv'}")  
    with open("mean_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Mean Time Taken", "Standard Deviation", "Standard Error of Mean", "Memory Mode", "Swarm Size", "Map"])  # write header
        for result in mean_results:
            writer.writerow(result)
    print("Mean values file saved successfully")  
    calculate_coefficient_variance()

def calculate_coefficient_variance():
    open('coefficient_variance_results.csv', 'w').close()
    # Load the data
    data = pd.read_csv('mean_results.csv')

    # Group data by Memory Mode and Swarm Size to calculate CV for each configuration across maps
    grouped_data = data.groupby(['Memory Mode', 'Swarm Size'])
    
    # Initialize list to store CV results
    cv_results = []

    # Iterate over each group
    for name, group in grouped_data:
        memory_mode, swarm_size = name
        # Calculate the mean and standard deviation across different maps for the same configuration
        overall_mean = np.mean(group['Mean Time Taken'])
        overall_std = np.std(group['Mean Time Taken'])
        # Calculate the coefficient of variance (CV)
        if overall_mean != 0:
            cv = overall_std / overall_mean
        else:
            cv = 0  # To handle cases where the mean might be zero
        # Append results to the list
        cv_results.append([memory_mode, swarm_size, cv])
    
    # Save the CV results to a new CSV file
    with open('coefficient_variance_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Memory Mode', 'Swarm Size', 'Coefficient of Variance'])  # Write header
        for result in cv_results:
            writer.writerow(result)
    
    print("Coefficient of variance results file saved successfully")


# Main function
def Main():
    global SCREEN
    # Ask the user to enter 'run' to start the simulation or 'build' to build the map
    start_command = input("Please enter 'run' to start the simulation, test to run experimints, or 'build' to enter map builder: ")
    if start_command.lower() == 'run':
        # Initialize Pygame here, inside this condition
        pygame.init()
        SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.DOUBLEBUF)
        testMap = '-1'
        swarmSize = '-1'
        memoryMode = '-1'
        while testMap not in ['1', '2', '3']:
            testMap = input("Please enter the test map (1, 2 or 3): ")
        while swarmSize not in ['50', '100', '150']:
            swarmSize = input("Please enter the swarm size (50, 100 or 150): ")
        while memoryMode not in ['0', '1', '2']:
            memoryMode = input("Please enter the memory mode (0 = Discovery, 1 = Pheromone, 2 = Both): ")
        
        
        runSimulation(int(memoryMode), int(testMap), int(swarmSize))
    elif start_command.lower() == 'test':
        pygame.init()
        SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.DOUBLEBUF)
        runExperiment()
    elif start_command.lower() == 'build':
        BuildMap()
    else:
        print("Invalid command. Please enter 'run' to start the simulation or 'build' to build the map.")


if __name__ == "__main__":
    Main()
    pygame.quit()