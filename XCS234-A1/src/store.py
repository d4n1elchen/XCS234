import copy
import random

import numpy as np


class StoreKeeper:
    def __init__(self, current, seed=1234):
        self.num_states = 11
        self.num_actions = 2  # O <=> buy, 1 <=> sell

        # Configure reward function
        R = np.zeros((self.num_states, self.num_actions))
        for s in range(1, self.num_states - 2):
            R[s, 0] = 1.0
        R[9, 1] = 100.0

        # Configure transition function
        T = np.zeros((self.num_states, self.num_actions, self.num_states))

        T[0, 1, 1] = 1.0

        # Terminal state
        T[10, 0, 9] = 0.0
        T[10, 1, 10] = 0.0

        for s in range(1, self.num_states - 2):
            T[s, 0, s - 1] = 1.0
            T[s, 1, s + 1] = 1.0

        self.R = np.array(R)
        self.T = np.array(T)

        # Agent always starts at the opposite end of the river
        self.init_state = 3
        self.curr_state = self.init_state

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_model(self):
        return copy.deepcopy(self.R), copy.deepcopy(self.T)

    def reset(self):
        return self.init_state

    def step(self, action):
        reward = self.R[self.curr_state, action]
        next_state = np.random.choice(
            range(self.num_states), p=self.T[self.curr_state, action]
        )
        self.curr_state = next_state
        return reward, next_state
