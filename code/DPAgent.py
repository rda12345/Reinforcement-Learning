#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPAgent is a class implementing two dynamical programming reinforcement learning
methods.
    
    1. Policy iteration - Initialize the state-value function, V, and the policy
                    arbitrarly. Perform iterations of sequential of 
                    iterative policy evaluation and policy improvement until
                    obtaining a stable policy (convergence).
                    iterative policy evaluation: V is computed for a 
                    the current policy utilizing an iterative method.
                    policy imporvement: for the current V check wether a better
                    policy exists.
    2. Value iteration - 
                        
The methods are utilized to find the optimal policy (or a proxy of that) in a 
finite Markovian Decision Process (MDP). Utilizing Bellman equation one can 
prove that the method converges for finite-MDPs.

The considered policy is deterministic, i.e., for every state there is a single
action.
"""
import numpy as np

class DPAgent(object):
    """The learning agent"""
    
    def __init__(self,
                 environment,
                 gamma: float = 0.95,
                 threshold = 1e-6):
        """
        Parameters
        ----------
        gamma : float, discount factor, dictates the importance of future reward (should be in the range [0,1)]) 
        threshold : float, the acceptable convergence error of the policy evaluation
        """
        # environment
        self.env = environment   
        self.base_env = self.env.unwrapped
        
        # discount factor
        self.gamma = gamma
        
        # convergence threshold
        self.threshold = threshold
        
        # number of states and actions
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        # initializing a random policy and value-state function
        self.policy = np.random.choice(self.n_actions,self.n_states)
        self.V = np.zeros(self.n_states)
        
    def get_V(self):
        """Returns the value-state function"""
        return self.V 
    
    def get_policy(self):
        """Returns the policy"""
        return self.policy
    
        
        
    def policy_evaluation(self):
        """Evaluates the state value function associated with the agent's policy"""
        Delta = self.threshold + 1
        while Delta >= self.threshold:
            Delta = 0
            for s in range(self.n_states):
                # store the previous value function
                prev_value_func = self.V[s]
                #print(f'previous V {prev_value_func}')
                # evaluate the updated state value function
                self.V[s] = self.update_V(s)
                # evaluate the improvment
                Delta = max(Delta,abs(prev_value_func - self.V[s])) 
            
    def update_V(self,s):
        """Evaluates the state-value function"""
        result = 0
        a = self.policy[s]
        for prob, next_state, reward, terminated in self.base_env.P[s][a]:                        
            result += prob * (reward + self.gamma * self.V[next_state])
        return result
    
    def policy_values(self, s, a):
        """
        Evaluates the policy value of a state s, action a and state-value 
        function V. The function is utilized by the policy improvement function.
        
        Parameters
        ----------
        s : int, representing a state in the environment's observation space
        a : int, reprenting an action in the action space
        """
        result = 0
        for prob, next_state, reward, terminated in self.base_env.P[s][a]:
            result += prob * (reward + self.gamma * self.V[next_state])
        return result
        
    def policy_improvement(self):
        """
        For a fixed state-value function, V, explores wether there are better
        actions. The local optimization improves the policy, pushing it closer
        to the optimal solution.
        """
        policy_stable = True 
        for s in range(self.n_states):
            a = self.policy[s]
            policies = np.array([self.policy_values(s,a) for a in range(self.n_actions)])
            self.policy[s] = np.argmax(policies)
            if a != self.policy[s]:
                policy_stable = False
        if policy_stable:
            return policy_stable
        
        
    
    def policy_iteration(self, max_iters):
        """
        Evaluates the optimal policy by policy iteration.
        Performs iterative sequential policy evaluation and policy improvement
        until the policy converges.
        
        Parameters
        ----------
        max_iters : int, maximum number of iterations. 
        """
        for _ in range(max_iters):        
            # policy evaluation
            self.policy_evaluation()
            
            # policy improvement
            policy_stable = self.policy_improvement
            if policy_stable:
                break
            
            
    def get_action(self,state):
        """
        Returns the agent's action for a certain environment state, i.e, policy[state]
        """
        return self.policy[state]
                 
    
    
    def value_iteration(self):
        """
        Evaluates the optimal policy with the value iteration method.
        """
        Delta = self.threshold + 1
        while Delta > self.threshold:
            Delta = 0
            for s in range(self.n_states):
                prev_value_func = self.V[s]
                # list containing the state-value function value for different actions
                values = []
                for a in range(self.n_actions):
                    values.append(self.policy_values(s, a))
                self.V[s] = max(values)
                Delta = max(Delta, prev_value_func - self.V[s])
        
        # For each state, s, evaluate the evaluate the action which maximizes
        # the policy_value function
        for s in range(self.n_states):
            policies = []
            for a in range(self.n_actions):
                policies.append(self.policy_values(s, a))
            self.policy[s] = np.argmax(np.array(policies))    
                        
        
        



    
    
    
    


