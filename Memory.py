import random
import numpy as np


class Memory():
    def __init__(self,max_len,batch_size):
        self.max_len = max_len
        self.current_index = -1
        self.counter = 0
        self.memory = [None for _ in range(self.max_len)]
        self.batch_size = batch_size

    def append(self, state, action, next_state, reward, done):
        self.current_index = (self.current_index + 1) % self.max_len
        self.memory[self.current_index] = (state, action, next_state, reward, done)
        self.counter += 1

    def sample(self):
        batch = random.sample(self.memory[:min(self.counter, self.max_len)], self.batch_size)
        batch = list(zip(*batch))

        state_batch = np.asarray(batch[0], dtype=np.float32)
        action_batch = np.asarray(batch[1],dtype=np.int)
        next_state_batch = np.asarray(batch[2], dtype=np.float32)
        reward_batch = np.asarray(batch[3], dtype=np.float32)
        done_batch = np.asarray(batch[4], dtype=np.int32)

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch