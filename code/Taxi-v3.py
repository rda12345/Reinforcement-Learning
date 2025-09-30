#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taxi-v3

Environment - it consists of a 5 by 5 grid world. There are four special cells, R, G, Y and B.
The aim of the taxi (the agent) is to pick up a passenger in one of the four 
special cells and drop them off in the correct destination.
The grid contain walls which the taxi cannot pass directly.
The possible actions are the move west, east, north, and south pickup and dropoff passenger.

Rewards - -1 per step to (to encourage efficiency),
          +20 for successful drop-off,
          -10 for illegal pickup/dropoff


Solution - solved with policy and value iteration methods.
"""



import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from DPAgent import DPAgent


## Model parameters
gamma: float = 0.95 # discount factor, dictates the importance of future reward (should be in the range [0,1)])
threshold: float = 1e-6 # the ac
max_iters: int = 10**3  # maximum iterations of the policy and value iterations.

## Initializing the agent and setting initial parameters
# create the Taxi environment
env = gym.make("Taxi-v3", render_mode="ansi")
agent = DPAgent(env, gamma, threshold)
# resent the environment (start a new episode)
state, info = env.reset() # state = (taxi_row, taxi_col, passenger_location, destination)
print(env.render())


## Training with policy iteration method
for _ in range(max_iters):
    # policy evaluation
    agent.policy_evaluation()
        
    # policy improvement
    policy_stable = agent.policy_improvement()
    if policy_stable:
        break
    


## Evaluation of the policy iteration method
actions = []
rewards = []
done = False
Return = 0
while not done:
    action = agent.get_action(state)   # returns policy[state]
    next_state, reward, terminated, truncated, _ = env.step(action)
    actions.append(action)
    Return += reward
    #dagent.update_Q(state,action,reward,next_state)
    done = terminated or truncated
    state = next_state
    


# Test Results
translator = {0: 'south', 1: 'north', 2: 'east', 3: 'west', 4: 'pickup', 5: 'dropoff'}
actions = [translator[a] for a in actions]
outcome = reward == 20
print('POLICY ITERATION TEST RESULTS:')
print(f'Successful Outcome: {outcome}')
print(f'Accumulated Reward: {Return}')
print(f'Actions sequence: {actions}')



## Evaluation of the value interation method

# Initialize agent (sets the value-state function and policy to be random)
agent = DPAgent(env, gamma, threshold)
state, info = env.reset()



## Training with policy iteration method
for _ in range(max_iters):
    # policy evaluation
    agent.value_iteration()
        
    # policy improvement
    policy_stable = agent.policy_improvement()
    if policy_stable:
        break
    

actions = []
rewards = []
done = False
Return = 0
while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    actions.append(action)
    Return += reward
    #dagent.update_Q(state,action,reward,next_state)
    done = terminated or truncated
    state = next_state
    
# Test Results
translator = {0: 'south', 1: 'north', 2: 'east', 3: 'west', 4: 'pickup', 5: 'dropoff'}
actions = [translator[a] for a in actions]
outcome = reward == 20
print('\n')
print('VALUE ITERATION TEST RESULTS:')
print(f'Successful Outcome: {outcome}')
print(f'Accumulated Reward: {Return}')
print(f'Actions sequence: {actions}')


env.close()





