################################### Imports ########################################
import pygame
import random
import math
from Map import *

# Initialize Pygame
pygame.init()

################################### Global Variables ########################################

# Simulation Constants
WINDOW_SIZE = 1000
GRID_SIZE = 500
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.DOUBLEBUF)
CLOCK = pygame.time.Clock()
MAXIMUM_DISTANCE = math.sqrt(GRID_SIZE**2 + GRID_SIZE**2)
MAX_ITERATIONS = math.inf

# PSO Constants
C1 = 1 # Cognitive coefficient
C2 = 1 # Social coefficient
INERTIA_WEIGHT = 0.5 # Inertia weight

# Swarm Constants
REOPTIMISATION_RATE = 5 # Reset the optimisation process every n iterations
NUMBER_OF_PARTICLES = 100 # Number of particles in the swarm
DISPERSE_RADIUS = 20 # Radius of disperse from other particles

MAX_VELOCITY = 10 # Maximum velocity of the particles
MAX_DISCOVERY = NUMBER_OF_PARTICLES # Maximum discovery value
MAX_PHEROMONES = NUMBER_OF_PARTICLES * 5 # Maximum number of pheromones before vaporisation
MEMGRID_SIZE = 0.1

# Particle States
STATE_SEEK = 0
STATE_ATTACK = 1
STATE_DEACTIVATE = 2





################################### Target Class ########################################

class Target:
    
    # Initialize the target
    def __init__(self, x, y):
        self.position = [x, y]
    
    # Check if the target is discoverable
    def isDiscoverable(self, px, py):
        distance = math.sqrt((px - self.position[0])**2 + (py - self.position[1])**2)
        return distance < 100
    
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
    def __init__(self, nParticles, c1, c2, inertiaWeight):
        self.mode = -1 # 0 = Discovery, 1 = Pheromone, 2 = Both
        self.nParticles = nParticles # Number of particles in the swarm
        self.c1 = c1 # Cognitive coefficient
        self.c2 = c2 # Social coefficient
        self.inertiaWeight = inertiaWeight # Inertia weight
        self.gBestPos = (0, 0) # Global best position
        self.gBestFitness = 0 # Global best fitness
        self.iCount = 0 # Iteration count
        self.memGrid = [[0] * int(GRID_SIZE * MEMGRID_SIZE) for _ in range(int(GRID_SIZE * MEMGRID_SIZE))] # Memory grid
        self.update = False # Flag to indicate if the memory grid has been updated
        self.pheromones = [] # List of pheromones placed by the particles
        self.nPheromones = 0 # Number of pheromones placed
        self.avrPheromonePos = (0, 0) # Average position of all pheromones
        self.prevPos = [] # List of previous positions of the leader
        self.avrPos = (0,0) # Average position of all particles
        self.leader = 0 # Index of the leader particle
    
    
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
        self.update = False
    
    
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
        self.id = id # Particle ID
        self.state = STATE_SEEK # Particle state
        self.position = [x, y] # Particle grid position
        self.memory = memory # Shared swarm memory
        self.velocity = [random.uniform(-MAX_VELOCITY, MAX_VELOCITY), random.uniform(-MAX_VELOCITY, MAX_VELOCITY)] # Particle velocity
        self.pBestPosition = [x, y] # Personal best position
        self.pBestFitness = 0 # Personal best fitness
        
        # Place a pheromone at the particle's initial position
        self.memory.placePheromone(self.position) 


    # Check if the particle is lost
    def checkLost(self):
        self.memory.calculateAvrPos()
        distance = math.sqrt((self.position[0] - self.memory.avrPos[0])**2 + (self.position[1] - self.memory.avrPos[1])**2)
        if distance > GRID_SIZE / 4:
            return True
    
    
    # Activate the particle
    def activate(self, grid, targets):
        if self.id == self.memory.leader:
            self.memory.calculateAvrPos()
            self.memory.recordPosition(self.position)
            if self.memory.mode == 1 or self.memory.mode == 2:
                self.memory.calculateAvrPheromone()
        
        # Check if the particle is lost
        if self.checkLost():
            self.state = STATE_DEACTIVATE
            if self.id == self.memory.leader:
                self.memory.leader += 1
            self.draw()
            return
        
        # default state is seek
        self.state = STATE_SEEK
        
        # Check if the particle is close to a target
        for target in targets:
            if target.isDiscoverable(self.position[0], self.position[1]):
                # If the particle is close to a target, switch to attack state
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
                self.memory.iCount % REOPTIMISATION_RATE == 0:
            self.memory.update = True
            if self.memory.mode != 1 and self.memory.memGrid[gridX][gridY] < MAX_DISCOVERY:
                self.memory.memGrid[gridX][gridY] += 1
            if self.memory.mode != 0:
                self.memory.placePheromone(self.position)
                
        # If the agent is in the attack state check if it has reached a target
        if self.state == STATE_ATTACK:
            for target in targets:
                if target.isAttacked(self.position[0], self.position[1]):
                    targets.remove(target)
                    self.state = STATE_SEEK
                    self.memory.restartOptimisation()
        
        # Restart the optimisation if the last particle has updated the memory grid or placed a pheromone
        if self.id == self.memory.leader:
            self.memory.iCount += 1
            if self.state == STATE_SEEK and self.memory.update == True:
                self.memory.restartOptimisation()
        self.memory.vaporise()

        # Draw the particle
        self.draw()


    # Disperse the particle from other particles
    def disperse(self, particles, grid):
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
                    self.velocity[0] += dx / distance  
                    self.velocity[1] += dy / distance
    
    
    # Calculate the fitness of the current position
    def calculateFitness(self, targets):
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
            
            # Ensure gridX and gridY are within the bounds of the memory grid            
            if gridX >= len(self.memory.memGrid):
                gridX = len(self.memory.memGrid) - 1
            if gridY >= len(self.memory.memGrid[0]):
                gridY = len(self.memory.memGrid[0]) - 1
            
            # Calculate the discovery amount and pheromone values
            discovery = self.memory.memGrid[gridX][gridY]
            discovery = 1 + (discovery / MAX_DISCOVERY)
            discovery = 2 - discovery
            distance = math.sqrt((self.position[0] - self.memory.avrPheromonePos[0])**2 + (self.position[1] - self.memory.avrPheromonePos[1])**2)
            pheromone = (distance / MAXIMUM_DISTANCE)
            
            # Calculate the fitness based on the mode
            if self.memory.mode == 0:
                fitness = discovery
            elif self.memory.mode == 1:
                fitness = pheromone
            elif self.memory.mode == 2:
                fitness = discovery + pheromone / 2
            else:
                fitness = random.uniform(0, 1)
                
        # Return the fitness
        return fitness


    # Update the velocity of the particle
    def update_velocity(self, gBestPos, inertiaWeight, c1, c2):
        # Calculate random values
        r1 = random.random()
        r2 = random.random()
        
        # Update the velocity using the PSO formula
        self.velocity[0] = inertiaWeight * self.velocity[0] + c1 * r1 * (self.pBestPosition[0] - self.position[0]) + c2 * r2 * (gBestPos[0] - self.position[0])
        self.velocity[1] = inertiaWeight * self.velocity[1] + c1 * r1 * (self.pBestPosition[1] - self.position[1]) + c2 * r2 * (gBestPos[1] - self.position[1])
        
        # Apply velocity clamping
        velocity_mag = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        while velocity_mag > MAX_VELOCITY or velocity_mag < -MAX_VELOCITY:
            # Scale down velocity to the maximum allowed
            velocity_mag = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            self.velocity[0] = (self.velocity[0] / velocity_mag) * MAX_VELOCITY 
            self.velocity[1] = (self.velocity[1] / velocity_mag) * MAX_VELOCITY


    # Update the position of the particle
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
                self.velocity[0] = -self.velocity[0]
            if proposed_y <= 0 or proposed_y >= GRID_SIZE - 1:
                self.velocity[1] = -self.velocity[1]

            # Update position with adjusted velocity
            self.position[0] += self.velocity[0]
            self.position[1] += self.velocity[1]
            self.position = [proposed_x, proposed_y]
        else:
            # Attempt to slide along the obstacle by checking alternative movements
            alternative_x = int(self.position[0] + self.velocity[0])
            alternative_y = int(self.position[1] + self.velocity[1])
            
            # Ensure alternative_x and alternative_y are within grid bounds
            alternative_x = max(0, min(GRID_SIZE - 1, alternative_x))
            alternative_y = max(0, min(GRID_SIZE - 1, alternative_y))
            
            if grid[alternative_x][int(self.position[1])] != 1:
                self.position[0] = alternative_x  # Move horizontally only
            if grid[int(self.position[0])][alternative_y] != 1:
                self.position[1] = alternative_y  # Move vertically only
        

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
def runSimulation():
    visualiseMemGrid = False
    visualisePrevPos = False
    visaulisePath = False
    visualiseMemory = False
    map = Map(GRID_SIZE, 20, 1, 8)
    spawn = map.loadSpawnPos()
    memory = SwarmMemory(NUMBER_OF_PARTICLES, C1, C2, INERTIA_WEIGHT)
    particles = []
    
    for i in range(NUMBER_OF_PARTICLES):
        particles.append(Agent(spawn[0], spawn[1], i, memory))

    memory.setParticles(particles)
    running = True
    iCount = 0
    targets = []
    targetPositions = map.loadTargetPositions()
    for pos in targetPositions:
        t = Target(pos[0], pos[1])
        targets.append(t)
    while running:
        iCount += 1
        if memory.mode == 0:
            visualiseMemGrid = True
            visualisePrevPos = False
        elif memory.mode == 1:
            visualisePrevPos = True
            visualiseMemGrid = False
        elif memory.mode == 2:
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
        

        for target in targets:
            target.draw()
        
        pygame.display.flip()
        CLOCK.tick(30)
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
                    memory.resetPrevMem()
                    memory.mode = 0
                if event.key == pygame.K_1:
                    memory.resetPrevMem()
                    memory.mode = 1
                if event.key == pygame.K_2:
                    memory.resetPrevMem()
                    memory.mode = 2
                if event.key == pygame.K_p:
                    visaulisePath = not visaulisePath
                    drawMemoryVisualisation(memory, WINDOW_SIZE, GRID_SIZE, visualiseMemGrid, visualisePrevPos, visaulisePath)
                if event.key == pygame.K_v:
                    visualiseMemory = not visualiseMemory
    pygame.quit()
    

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