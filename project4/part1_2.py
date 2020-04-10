# part1_2.py: Project 4 Part 1 script
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
from mdp_env import mdp_env
from mdp_agent import mdp_agent

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    policy = np.empty_like(env.states)
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    iter = 0
    while True:
        iter += 1
        delta = 0
        U = agent.utility.copy()
        for s in range(len(env.states)):        # for each state s in S do the following
            max_sum = 0
            for a in range(len(env.actions)):
                temp_sum = 0
                for s_prime in range(len(env.states)):
                    temp_sum += env.transition_model[s, s_prime, a] * U[s_prime]
                if temp_sum > max_sum:
                    max_sum = temp_sum
            # update utility function
            agent.utility[s] = env.rewards[s] + agent.gamma * max_sum
            # update delta
            if abs(agent.utility[s] - U[s]) > delta:
                delta = abs(agent.utility[s] - U[s])

        # stay in while loop until this condition
        if delta < (eps * (1 - agent.gamma) / agent.gamma):
            break
        # or has reached the maximum iteration
        if iter == max_iter:
            break

    # determine the policy based on the optimal utility function
    for i in range(len(policy)):
        opt_a = 0
        max_sum = 0
        # find the optimal action that maximizes the expected utility
        for a in range(len(env.actions)):
            temp_sum = 0
            for s_prime in range(len(env.states)):
                temp_sum += env.transition_model[i, s_prime, a] * agent.utility[s_prime]
            if temp_sum > max_sum:
                max_sum = temp_sum
                opt_a = a
        policy[i] = opt_a
    ## END Student code
    return policy

