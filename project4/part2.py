# part2.py: Project 4 Part 2 script
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
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    # np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states)))
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    iter = 0
    while iter < max_iter:
        iter += 1
        policy_evaluation(env, agent, 20, policy)
        unchanged = True
        for s in range(len(env.states)):
            max_sum = 0
            opt_a = 0
            for a in range(len(env.actions)):
                temp_sum = 0
                for s_prime in range(len(env.states)):
                    temp_sum += env.transition_model[s, s_prime, a] * agent.utility[s_prime]
                if temp_sum > max_sum:
                    max_sum = temp_sum
                    opt_a = a

            # policy_sum = 0
            # for s_p in range(len(env.states)):
            #     policy_sum += env.transition_model[s, s_p, policy[s]] * agent.utility[s_p]
            #
            # if max_sum > policy_sum:
            #     policy[s] = opt_a
            #     unchanged = False

            if opt_a != policy[s]:
                policy[s] = opt_a
                unchanged = False
        if unchanged:
            break
    ## END: Student code
    return policy


def policy_evaluation(env: mdp_env, agent: mdp_agent, k, policy):
    for i in range(k):
        for s in range(len(env.states)):
            util_sum = 0
            for s_prime in range(len(env.states)):
                util_sum += env.transition_model[s, s_prime, policy[s]] * agent.utility[s_prime]
            agent.utility[s] = env.rewards[s] + agent.gamma * util_sum