# part1_1.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2020
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Dr. Jonathan Kelly
# jkelly@utias.utoronto.ca
#
# Teaching Assistant:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca


###
# Imports
###

import numpy as np
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleanign robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])
    # P.shape = (6, 6, 2)

    ## START: Student Code
    for s1 in range(P.shape[0]):
        if s1 == 0:
            P[s1][s1][1] = 0            # terminal states: once reached, stay
            P[s1][s1][0] = 0            # no matter what the action is
        elif s1 == P.shape[0] - 1:
            P[s1][s1][1] = 0            # terminal states: once reached, stay
            P[s1][s1][0] = 0            # no matter what the action is
        else:
            left = s1 - 1
            right = s1 + 1
            P[s1][right][1] = 0.8       # action: right; s2: right
            P[s1][s1][1] = 0.15         # action: right; s2: s1 (stay)
            P[s1][left][1] = 0.05       # action: right; s2: left
            P[s1][right][0] = 0.05      # action: left; s2: right
            P[s1][s1][0] = 0.15         # action: left; s2: s1 (stay)
            P[s1][left][0] = 0.8        # action: left; s2: left
    ## END: Student code
    return P

