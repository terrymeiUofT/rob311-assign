import random
import gym
import math
import numpy as np

"""
Agent Description:

The Reinforcement Learning strategy being used to train this CartPoleAgent is 
the REINFORCE policy gradient method. The major distinction of this method 
from other RL strategies is its weight update frequency: REINFORCE method 
updates the weight after the terminal state is reached in an episode.

Some of the key techniques utilized in this method is listed below:
- The probability distribution for each action (L or R) is calculated using 
  exponential soft-max function. 
- At each step, an action is chosen randomly with the guidance of a 
  probability distribution. This distribution is the same within each episode.
- The weight is a function of the gradient at each step and the  discounted 
  future reward[1](equation 13.8). The techniques for computing gradients of 
  the policy is found from this source[2]. The reward function is biased 
  towards the action taken in near future.

References:
[1]R. Sutton and A. Barto, Reinforcement learning.
[2]S. Kirkiles, "REINFORCE Policy Gradients From Scratch In Numpy", Medium, 2018
   [Online]. Available: https://medium.com/samkirkiles/reinforce-policy
   -gradients-from-scratch-in-numpy-6a09ae0dfe12. [Accessed: 13-Apr-2020].
"""


class CartPoleAgent:

    def __init__(self, observation_space, action_space):
        # ----- TODO: Add your code here. -----
        # Store observation space and action space.
        self.observation_space = observation_space
        self.action_space = action_space
        # added fields
        self.weight = np.random.rand(4, 2)
        self.action_prob = None
        self.grad_history = []
        self.reward_history = []
        self.gamma = 0.99
        self.lr = 0.001
        self.change_lr = False
        self.last_20_epi_reward = []

    def action(self, state):
        """Choose an action from set of possible actions."""
        # ----- TODO: Add your code here. -----
        # reshape state to (4, 2) to match the shape of the weight
        state = state[None, :]

        # compute the prob. of taking each action using softmax
        dot_product = state.dot(self.weight)[0]
        softmax_numer = np.exp(dot_product)
        softmax_denom = np.sum(softmax_numer)
        self.action_prob = softmax_numer / softmax_denom

        # choose the action randomly given the probability distribution
        action = np.random.choice(self.action_space.n, p=self.action_prob)
        # print("Chose action " + str(action))
        return action

    def reset(self):
        """Reset the agent, if desired."""
        # ----- TODO: Add your code here. -----
        self.grad_history = []
        self.reward_history = []

    def update(self, state, action, reward, state_next, terminal):
        """Update the agent internally after an action is taken."""
        # ----- TODO: Add your code here. -----
        # compute the Jacobian matrix for softmax
        reshaped_prob = self.action_prob.reshape(2, 1)
        Jacob_mat = np.diagflat(reshaped_prob) - \
                    reshaped_prob.dot(np.transpose(reshaped_prob))

        # find the gradient of log policy
        d_policy = Jacob_mat[action, :]
        policy = self.action_prob[action]
        d_log_policy = d_policy / policy
        gradient = (np.transpose(state).reshape(4, 1)).dot(
            d_log_policy.reshape(1, 2))

        # update the history of gradient and reward
        self.grad_history.append(gradient)
        self.reward_history.append(reward)

        # update the weight if reach terminal
        if terminal:
            for t in range(len(self.grad_history)):
                accum_reward = 0
                for i in range(t, len(self.reward_history)):
                    accum_reward += (self.gamma ** (i-t)) * \
                                    (self.reward_history[t])
                self.weight += self.lr * self.grad_history[t] * accum_reward

            # lower the learning rate if reward is constantly high
            episode_reward = sum(self.reward_history)
            if len(self.last_20_epi_reward) < 20:
                self.last_20_epi_reward.append(episode_reward)
            else:
                self.last_20_epi_reward.pop(0)
                self.last_20_epi_reward.append(episode_reward)

            if self.change_lr:
                pass
            else:
                if len(self.last_20_epi_reward) == 20:
                    high_count = 0
                    for i in range(20):
                        if self.last_20_epi_reward[i] > 300:
                            high_count += 1
                    if high_count > 15:
                        self.change_lr = True
                    if self.change_lr:
                        self.lr = 0.0001
