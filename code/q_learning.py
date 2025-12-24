#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-learning is a model-free reinforcement learning algorithm that teaches an agent
to choose actions that maximize long-term rewards, without needing to know the
environment’s dynamics.
The central object is the lookup table:  Q(s,a) estimating the value
(expected future reward) of taking action a in state s. The method learns by
trail and error, while exploring the environment.

Procedure:
    1. The adjent observes the environment state s
    2. Takes and action a
    3. Recieves a reward r
    4. Environment state is modified to s'
    
The method is based on the Bellman Optimality Equation that
says that the true optimal Q-value satisfies: E[reward + gamma*max_{a'}(Q[s',a'])].
"""
import numpy as np

class QLearnAgent(object):
    """The learning agent"""
    
    def __init__(self,
                 environment,
                 alpha: float = 0.8,
                 gamma: float = 0.95,
                 initial_epsilon: float = 1.0,
                 epsilon_decrease: float = 0.99,
                 final_epsilon: float = 0.4):
        """
        Parameters
        ----------
        alpha : float, learning rate
        gamma : float, discount factor, dictates the importance of future reward (should be in the range [0,1)]) 
        initial_epsilon : float, exploration rate
        epsilon_decrease : float, decrease rate of the exploration rate
        final_epsilon : float, final exploration rate
        Q : np.ndarray (num of state, num of actions), the Q-table
        training_error : list, stores the training errors
        """
        # environment
        self.env = environment        
        
        # learning rate
        self.alpha = alpha   
        
        # discount factor
        self.gamma = gamma    
        
        # exploration rate
        self.epsilon = initial_epsilon  
        self.epsilon_decrease = 0.99
        self.final_epsilon = 0.4
        
        # Q-table
        self.Q = np.zeros((self.env.observation_space.n,self.env.action_space.n))       
        
        # training error
        self.training_error = []
        
    def get_action(self,state):
        """
        Parameters 
        ----------
        state: int, current state
        
        Returns
        -------
        Agent's action
        """
        # Explore the action space with epsilon probability
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        # or take the best estimated action
        else:
            action = np.argmax(self.Q[state,:])
        return action
                 
    
    def update_Q(self,state,action,reward,next_state):
        """
        Updates Q(s,a) according to the update rule.        
        
        Parameters 
        ----------
        state : int, current state
        action : int, action taken by the agent
        reward : int, reward recieved by the agent
        next_state : int, environment's next state        
        """
        # difference between the estimated target and the current state
        temporal_difference = reward +\
            self.gamma*(np.max(self.Q[next_state,:])) - self.Q[state,action]
        self.Q[state,action] = self.Q[state,action] + self.alpha*(temporal_difference)
        
        # Track the training error for analysis
        self.training_error.append(temporal_difference)
        
    
    def decrease_epsilon(self):
        """
        Returns the exploration rate, epsilon, which each episode is decreased
        by a factor of decrease_epsilon.    
        """
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decrease)
        return self.epsilon
    
        



    
    
    
    


