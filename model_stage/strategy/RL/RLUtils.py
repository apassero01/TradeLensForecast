from random import sample

import numpy as np


class TradingEnv:
    def __init__(self, x, close_price):
        self.x = x
        self.close_price = close_price
        self.action_space = [0,1,2]
        self.n_samples = x.shape[0]
        self.current_index = 0
        self.position = None
        self.actions = []

    def reset(self):
        self.current_index = 0
        self.position = None
        self.actions = []
        return self.x[self.current_index]

    def step(self, action):
        if self.current_index >= self.n_samples - 1:
            raise Exception("End of data reached")
        current_close = self.close_price[self.current_index]

        reward = 0.0

        if action == 0: # buy
            if self.position is None:
                self.position = current_close
            self.actions.append(0)
        elif action == 1: # sell
            if self.position is not None:
                reward = current_close - self.position
                self.position = None
            self.actions.append(1)
        elif action == 2: # hold
            reward += 0.0
            self.actions.append(2)
        else:
            raise Exception("Invalid action")

        self.current_index += 1
        done = self.current_index >= self.n_samples - 1
        next_state = self.x[self.current_index] if not done else None
        return next_state, reward, done, {}


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def epsilon_by_frame(frame_idx, epsilon_start = 1, epsilon_final=.1, epsilon_decay = 500):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)




