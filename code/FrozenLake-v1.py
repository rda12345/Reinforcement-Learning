#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrozenLake-v1  

Enviornment - The Frozen lake environment describes a 4 by 4 grid of frozen cells
and holes. The agents begings on the starting cell, S (most left upper cell), and 
aims to travel to the final cell, G (most lower right cell), by avoiding the holes.
The agent can move only left, down, right and up.

Rewards - +1 for a successful solution. 

Solution - Applying Q-learning with a sufficiently high exploration rate leads
to an optimal policy. 

Note: successful learning process requires the exploration rate, epsilon to be
large enough for sufficiently long time.
"""

import numpy as np
import gymnasium as gym
from q_learning import QLearnAgent
import matplotlib.pyplot as plt


## Model parameters
alpha: float = 0.8  # learning rate
gamma: float = 0.95 # discount factor, dictates the importance of future reward (should be in the range [0,1)])
initial_epsilon: float = 1.0    # exploration rate
epsilon_decrease: float = 0.99  # decrease rate of the exploration rate
final_epsilon: float = 0.1      # final exploration rate



## Initializing the agent and setting initial parameters
env = gym.make("FrozenLake-v1",is_slippery=False)

agent = QLearnAgent(env,alpha,gamma,initial_epsilon,epsilon_decrease,final_epsilon)
episodes = 2*10**3      # Number of games played
returns = []            # Tracking the accumulated reward during a single episode
episode_length = []     # The total number of actions in a single episode
## Training
for i in range(episodes):
    # Reseting the environment, state is the current position of the agent
    state, _ = env.reset()
    done = False
    episode_return = 0   # Accumulated reward
    counter = 0         
    while not done:
        # Pick an action using an epsilon-greedy strategy
        action = agent.get_action(state)        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        # Update Q-table
        agent.update_Q(state,action,reward,next_state)            
        counter += 1
        done = terminated or truncated
        state = next_state
    # Store the return and the episode length      
    returns.append(episode_return)
    episode_length.append(counter)
## Analyzing training results
def get_moving_avgs(arr, window, convolution_mode):
    """Smooths noise by computing moving average"""
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window

# Smooth the temporal difference
rolling_length = 10
# Average reward, episode length and training error
reward_moving_avrg = get_moving_avgs(returns, rolling_length, "valid")
episode_length_moving_avrg = get_moving_avgs(episode_length, rolling_length, "valid")
training_error_moving_avrg = get_moving_avgs(agent.training_error, rolling_length, "valid")

fig, axs = plt.subplots(ncols=3, figsize = (12,3))
# Plot episode rewards
axs[0].set_title("Episode rewards")
axs[0].plot(range(len(reward_moving_avrg)),reward_moving_avrg)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Plot the episode length
axs[1].set_title("Episode lengths")
axs[1].plot(range(len(episode_length_moving_avrg)),episode_length_moving_avrg)
axs[1].set_ylabel("Average Reward")
axs[1].set_xlabel("Episode")
     

# Plote the training error
axs[2].set_title("Episode Training Error")
axs[2].plot(range(len(training_error_moving_avrg)),training_error_moving_avrg)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Steps")
plt.show()

# Test
episodes = 1
total_reward = 0
episode_length = []
actions = []
for i in range(episodes):
    state, _ = env.reset()
    done = False
    episode_return = 0
    counter = 0
    while not done:
        action = np.argmax(agent.Q[state,:])
        next_state, reward, terminated, truncated, _ = env.step(action)
        actions.append(action)        
        episode_return += 1
        #dagent.update_Q(state,action,reward,next_state)
        counter += 1
        done = terminated or truncated
        state = next_state
        total_reward += reward
    episode_length.append(counter)
        
env.close()

# Test Results
translator = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
actions = [translator[a] for a in actions]
print(f'Success percentage: {total_reward/episodes*100} %')  
print(f'Actions sequence: {actions}')
print('TEST RESULTS:')
#print(f'Q-table: {agent.Q}')  
#print(f'Max. episode length: {max(episode_length)}')

        
    
    



    