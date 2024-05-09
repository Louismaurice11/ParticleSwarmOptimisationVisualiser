# Swarm Simulation Project

This Python project simulates particle swarm optimization (PSO) to explore grid environments, avoid obstacles, and locate targets. The simulation employs a particle-based approach where agents (particles) disperse, avoid obstacles, and coordinate based on pheromone signaling or spatial discovery.

## Features

- **Dynamic Swarm Simulation**: Leverages the principles of PSO for dynamic navigation and target discovery.
- **Multiple Modes**: Supports different modes of operation including discovery, pheromone tracking, and a hybrid mode.
- **Obstacle Avoidance**: Includes mechanisms for agents to detect and avoid obstacles within the grid.
- **Performance Metrics**: Records simulation times and efficiency, outputting detailed statistics for analysis.
- **Interactive Visualizations**: Provides real-time visual feedback on agent states and paths, with adjustable parameters for detailed analysis.

## Installation

To run this simulation, you will need Python 3.x and the following packages:
- `pygame` for rendering the simulation
- `numpy` for numerical operations
- `scipy` for statistical functions
- `pandas` for data manipulation
- `matplotlib` for plotting (optional)

Install the required packages using pip:
 
```bash
pip install pygame numpy scipy pandas matplotlib
```

## Usage
To start the simulation, clone the repository and run the main.py file:
```bash
python main.py
```