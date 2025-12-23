#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Easy21

Implementation of the the easy21 environment, a simplified version of blackjack.
The rules of the game are described in detail in easy21-assignment.pdf
"""
import numpy as np
from gymnasium import spaces
class Easy21(object):
    
    def __init__(self):
        """
        Initializes the Easy21 environment
        
        Parameters
        ----------
        
        player_sum : int, sum of the player cards
        dealer_sum : int, sum of dealer cards
        dealer_first_card : int, dealer's first card
        action_space: tuple, containing the dimensions of the action space
        observation_space: tuple, containing the dimensions of the observation space
        stick: bool, a flag which indicates if the player sticks (chose to not take any more cards)
        turncated: None, a dummy parameter to fit to OpenAI's Gymnasium's package
        info: None, a dummy parameter to fit to OpenAI's Gymnasium's package
        
        """
        
        self.player_sum = 0
        self.dealer_sum = 0
        self.dealer_first_card = 0 
        self.action_space = spaces.Discrete(2)  # stick or hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # player sum
            spaces.Discrete(11)  # dealer card
            ))
        # dummy variables so the environment fits to the template of OpenAI's Gymnasium
        self.truncated = None  
        self.info = None
        
        
    def reset(self):
        """
        Resets the card game (environment)
        """
        self.dealer_first_card = np.random.choice(np.arange(1,11))
        self.player_sum = np.random.choice(np.arange(1,11))
        self.dealer_sum = self.dealer_first_card
        return (self.player_sum, self.dealer_first_card), self.info  # returns the state
        
        
        
    def _draw_card(self):
        # samples an integer in the range (1,10) uniformly
        number = np.random.choice(np.arange(1,11))
        # samples 'black' or 'red' with probabilities (2/3, 1/3), respectively
        color = np.random.choice([1, -1], p=[2/3, 1/3])
        return number, color
    
    def _reward(self, terminated):
        """
        Returns the the terminal reward
        """
        # reward is only given once the episode terminates (game finishes)
        if not terminated:
            return 0
        elif self.dealer_sum > 21 or self.dealer_sum < 1:
            return 1               # the dealer went bust
        elif self.player_sum > 21 or self.player_sum < 1:  # the player went bust
            return -1
        # both players didn't go bust 
        elif self.player_sum == self.dealer_sum:  # its a tie
            return 0
        else:
            return 2*int(self.player_sum > self.dealer_sum) - 1 # returns 1 and -1 if player won or lost, correspondingly
            
            
    def _is_terminated(self, action, dealer_action):
        """
        Checks if the episode terminated
        """
        if self.dealer_sum >= 21 or self.player_sum >= 21 or self.player_sum < 1 or self.dealer_sum < 1:
            return True
        if action==0 and dealer_action==0:  # both playes 'stick', the episode terminates
            return True
        return False
        
    def step(self, action):
        """
        Takes a player's action (hit: 1 or stick: 0)
        and returns the next state (which may be a terminal state), reward
        and wether the episode terminated.
        
        Parameters
        ----------
        action : int, 1 corresponds to hit (take another card) and 0 to stick
                    (don't take another card)
        
        Returns
        -------
        next_state : tuple, containing the player sum and dealer's first card
        reward : int, in the range (-1, 0, +1)
        terminated: bool, flags if the player of dealer sums are beyond 21.
        """
        # if the player hits, draw card
        if action:
            num, sign = self._draw_card()
            self.player_sum += sign * num
        
        
        # check if the player didn't go bust
        dealer_action = 0   # flags that the dealer's action
        if self.player_sum <= 21:
            # if the dealer's sum is bellow 17, hit
            if self.dealer_sum < 17:
                dealer_action = 1
                num, sign = self._draw_card()
                self.dealer_sum += sign * num

        # check if the episode is terminated
        terminated = self._is_terminated(action, dealer_action)
        # evaluate reward
        reward = self._reward(terminated)
        next_state = (self.player_sum, self.dealer_first_card)
        return (next_state, reward, terminated, self.truncated, self.info)
            
        
            
        
        
        
            
        
        
      