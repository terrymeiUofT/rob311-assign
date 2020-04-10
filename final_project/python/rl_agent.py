import random
import gym
import math
import numpy as np

"""
Agent Description:

TODO: Insert a 10-15 line description (80 characters wide) of the algorithm
that you implemented in your agent. If you used a reference paper or book,
you may add additional lines to cite this reference.
"""

class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        #----- TODO: Add your code here. -----

        # Store observation space and action space.
        self.observation_space = observation_space
        self.action_space = action_space

    def action(self, state):
        """Choose an action from set of possible actions."""
        #----- TODO: Add your code here. -----

        # Dummy agent just takes random actions...
        action = random.randint(0, self.action_space.n - 1)
        #print("Chose action " + str(action))
        return action

    def reset(self):
        """Reset the agent, if desired."""
        #----- TODO: Add your code here. -----
        pass

    def update(self, state, action, reward, state_next, terminal):
        """Update the agent internally after an action is taken."""
        #----- TODO: Add your code here. -----
        pass
