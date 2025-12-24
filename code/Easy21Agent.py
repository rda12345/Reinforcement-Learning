#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent is a class implementing a number of Monte Carlo (MC) reinforcement
learning methods:
    1.First-visit MC method for estimating the state-value function for a
        given policy.
    2.  On-policy first-visit Monte Carlo control algorithm with epsillong greedy policy. 
        Various possible update rules for the learning and exploration rates are proposed.
        

    
                        
The methods are utilized to find the optimal policy (or a proxy of that) in a 
finite Markovian Decision Process (MDP). 
"""
import numpy as np
from collections import defaultdict

class Easy21Agent(object):
    """The learning agent"""
    
    def __init__(self, 
                 environment,
                 max_iters: int = int(1e5),
                 policy = None,
                 alpha = 0.2,
                 epsilon = 0.4,
                 gamma = 0.9,
                 N0 = 100):
        """
        Parameters
        ----------
        environment : gymnasium environment
        max_iters :  int, maximum number of iterations
        policy: np.ndarray, a preset policy
        alpha: float, learning rate
        epsilon: float, exploration rate
        gamma: float, discount factor,  0<gamma<=1.0. 
        N0: int, parameter dictating the change in the exploration rate epsilon.
        n_spaces : tuple, containing the dimensions of the 
                (player sum, dealer's showing card) state space, 
                 defining the state.
        n_actions : int, number of actions
        policy : np.ndarray, deterministic policy, i.e., action for each state,
                        where the state is deterimed by a 3 element tuple
        V : np.ndarray, state-value function, expected return for every state
        Q : np.ndarray, action-value function, expected return for every state-action pair
        Returns : dict, maps the state tuple to a list of returns
        Q_Returns : dict, maps a tuple (state, action) to a list of returns
        """
        self.env = environment  # environment        
        self.max_iters = max_iters         # maximm number of iterations
        
        # descrite observation space assumed
        self.n_spaces = tuple([space.n for space in self.env.observation_space]) # state space size
        self.n_actions = self.env.action_space.n  # action space size
        self.actions = list(range(self.n_actions))
        
        # deterministic random policy
        if policy is not None:
            self.policy = policy
        else:
            self.policy = np.random.choice(self.n_actions, size=self.n_spaces)
        
        # action-value function
        self.Q = defaultdict(int)

        # state-value function        
        self.V = defaultdict(int)
        
        # learning and exploration rates
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.N0 = N0 # constant incorporated in the time-varying exploration rate
        
        # discount rate
        self.gamma = gamma
       
        
        # counters and returns storage
        self.Ns = defaultdict(int) # state visit counts
        self.Nsa =  defaultdict(int) # state-action pair visit counts
        
        
        # accumulated return dictionary: maps states to a list containing
        # total_return, number or returns.
        self.accumulated_return = {}

        
# ------------------------------ getters / setters ----------------------------        
        
    def get_V(self):
        """Returns the value-state function as a numpy array"""       
        V = np.zeros(self.n_spaces)
        for s in self.V.keys():
            player_sum, dealer_card = s
            V[player_sum, dealer_card] = self.V[s]
        return V 
        
        
    
    def get_Q(self):
        """Returns the action-value function"""
        Q = np.random.rand(*(*self.n_spaces,self.n_actions))
        for pair in self.Q.keys():
            s, a = pair
            player_sum, dealer_card = s
            Q[player_sum, dealer_card, a] = self.Q[(*s,a)]
        return Q
    
    
    def set_policy(self,policy):
        self.policy = policy
    
    def get_policy(self):
        """Returns the policy"""
        return self.policy
    
    
    def get_action(self,state, exploration_method = 'constant-eps'):
        """
        Parameters: 
            state: int, current state
            exploration_method: str, exploriation method, i.e., the update rule for epsilon
        Returns:
            Agent's action
            
        """ 
        assert(exploration_method == 'constant-eps' or exploration_method == 'varying-eps')
           
        
        if exploration_method == 'varying-eps':
            self.update_epsilon(state)
        if np.random.rand() < self.epsilon: 
            action = self.env.action_space.sample()  
        # or take the best estimated action by a greedy policy
        else:
            q_vals = [self.Q[(state, a)] for a in self.actions]
            max_actions = [a for a in self.actions if self.Q[state, a]==max(q_vals)]
            action = np.random.choice(max_actions)
        return action
        

        
# ---------------------------- MC policy evaluation ---------------------------      

    def MCpolicy_evaluation(self):
        """
        First-visit MC method for estimating the state-value function for a
        given policy.
        """
        returns = {}

        for _ in range(self.max_iters):
            # reset the list of episode (state, action, reward) tuples tuples
            episode = []
            
            # reset environment and choose the initial state
            state, _ = self.env.reset()    # (player sum, dealer's showing card, usable ace)
            done = False
            while not done:
                       
                # perform the next action
                action = self.policy[state]
                
                # for the chosen action observe the new environment state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                #store the triple
                episode.append((state, action, reward))
                        
                # reset the state
                state = next_state

                # check if episode is done
                done = terminated or truncated
            
            G = 0
            T = len(episode)
            G_list = [0.0] * T
            # Compute G backwards
            for t in reversed(range(len(episode))):
                _, _, reward_t = episode[t]
                G = self.gamma * G + reward_t
                G_list[t] = G
            
            visited = set()  # to insure first-visit
            
            # Add returns by a forward pass
            for t in range(T):    
                state_t, _, _ = episode[t]
                # First-visit check
                if state_t not in visited:
                    visited.add(state_t)
                    
                    if state_t not in returns:
                        returns[state_t] = []
                    returns[state_t].append(G_list[t])
            
        
        for state, G_list in returns.items():
            self.V[state] = sum(G_list)/len(G_list)    

# ------------------------------ Monte Carlo control --------------------------      
            
    def reset_record(self):
        self.Returns = defaultdict(int)
        self.Ns = defaultdict(int) # state visit counts
        self.Nsa = defaultdict(int)   # pair counter

        
    def MCcontrol(self,
                  learning_method='average',
                  exploration_method='varying-eps'):
        """
        On-policy first-visit Monte Carlo control algorithm with epsillong greedy policy. 
        Various possible update rules for the learning and exploration rates are proposed.
        
        
        
        Parameters:
            
            learning_method: str, sets the update rule.
                        'average' uses a simple average;
                        'constant-alpha' uses a given alpha.
                        'varying'uses a  learning rate which is inverse of the 
                        the number of time the state-action pair was encountered
                        
            exploration_method: str, 'constant-eps' uses a fixed epsilon:
                'varying-eps' uses N0/(N0+Ns[s]) where Ns[s] is the number times state s
                has been visited.
        
        If 'constant-alpha' chosen, pass alpha (float). Otherwise ignore alpha param.

        """
        assert(learning_method in ('average', 'constant-alpha', 'varying'))
        assert(exploration_method in ('constant-eps', 'varying-eps'))
        

        #actions = self.env.action_space
        for _ in range(self.max_iters):
            # initializing the step counter
            #counter = 0
            
            episode = []
            
            
            # reset environment and choose the initial state
            state, _ = self.env.reset()
            for state_1, action_1 in self.Q.keys():
                if not isinstance(state_1, tuple):
                    print('new')  
            action = 1
            done = False
            
            # Run an episode
            while not done:
                 
                self.Ns[state] += 1
                
                # get action if the player didn't stick
                if action:
                    action = self.get_action(state, exploration_method = exploration_method)
                    
                
                # for the chosen action observe the new environment state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode.append((state, action, reward))
                # reset the state
                state = next_state

                # check if episode is done
                done = terminated or truncated
            
 
            # Compute returns G_t backwards
            T = len(episode)
            G_list = [0.0]*T
            G = 0
            for t in reversed(range(T)):
                _, _, reward_t = episode[t]
                G = self.gamma * G + reward_t
                G_list[t] = G
                
            
            seen_pairs = set()  # to insure first-visit
            # Add returns by a forward pass
            for t in range(T):    
                state_t, action_t, _ = episode[t]
                # First-visit check
                pair = (state_t, action_t)

                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)  
                
                
                # update the pair counter, updated only once per first-visit update
                self.Nsa[pair] += 1
                Gt = G_list[t]
                if learning_method == 'varying':
                    #if not isinstance(state_t, tuple):
                        #print('check')
                    
                    self.Q[pair] = self.varying_update_rule(pair, Gt=Gt)
                elif learning_method == 'average': 
                    self.Q[pair] = self.average_update_rule(pair, Gt=Gt)
                elif learning_method == 'constant-alpha':
                    self.Q[pair] = self.constant_alpha_update_rule(pair, Gt=Gt)
                else:  NameError('Enter a valid update rule') 
 
            # update the policy
            self.update_policy(episode)   

         
        # evaluate the state-value function    
        self.update_V()
          
    def update_V(self):
        states = [s for s, _ in self.Q.keys()]
        states = set(states)  # removing duplicates
        #print(states)
        for s in states:
            a = self.policy[s]
            pair = (s,a)
            self.V[s] = self.Q[pair]
        
    def update_policy(self, episode):
        states = {s for (s, _, _) in episode}
        for s in states:
            q_vals = [self.Q[(s, a)] for a in self.actions]  
            max_q = max(q_vals)
            max_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
            self.policy[s] = np.random.choice(max_actions)          
            
    def average_update_rule(self,pair,Gt):
        return ((self.Nsa[pair]-1)/self.Nsa[pair]) * self.Q[pair] + (1/self.Nsa[pair]*Gt)
    
    
    def varying_update_rule(self,pair, Gt):
        self.alpha = 1/self.Nsa[pair]
        temporal_difference = Gt - self.Q[pair]
        return self.Q[pair] + self.alpha*(temporal_difference)
    
    def constant_alpha_update_rule(self,pair, Gt):
        temporal_difference = Gt - self.Q[pair]
        return self.Q[pair] + self.alpha*(temporal_difference)
    
    def update_epsilon(self,state):
        # Explore the action space with epsilon probability
        self.epsilon = self.N0/(self.N0 + self.Ns[state])
        
        




    
    
    
    


