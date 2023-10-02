# ----------------------------
# RoadAI-project
# Project for in5490
# ----------------------------

## --------
## Make sure you have installed a programming environment.
Options: venv, conda, poetry.
We have used venv.

### Installation
python -m venv venv

### Open environment
source env/bin/activate

### install required packets
pip install -r requirements.txt

## --------
## Make sure you have a config system installed.
We'll be using Hydra
Full tutorial: https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/

### Installation
pip install hydra-core --upgrade

## --------
## Class Overview
### TruckAgent class
#### __init__(self, pos_y, pos_x, ground, holes, agent_num):
Initializes the agent with its initial position, ground type, capacity, and other relevant parameters. It also creates an initial tensor to store agent-related information.

#### _in_bounds(self, pos, env):
A helper method that checks if a given position is within the bounds of the environment.

#### step(self, map, act_space, env):
Represents a single time step of the agent's behavior. It includes logic for random movement, collision detection, and cargo handling. The agent can move in random directions and interact with its surroundings.

#### deep_step(self, env, action):
Similar to the step method, but allows external control of the agent's actions. The agent can move up, down, left, or right based on the provided action.


## --------
## [Subsection]
[something here]

### [subsubsection]


