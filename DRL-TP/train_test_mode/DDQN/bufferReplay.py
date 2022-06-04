"""
function:Memory
date    :2021/8/25
author  :HLQ
"""
import random
import numpy as np

#=======================================#
#             Buffer Memory             #
#=======================================#
class MemoryBuffer:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.buffer = []
        self.nodes = 14


    def can_sample(self, batch_size):
        """
            Determine whether the data can be retrieved
        :param batch_size:
        :return:
        """
        return len(self.buffer) >= batch_size


    def can_sample_to_rnn(self, batch_size):
        """
            GRU's batch.
        :param batch_size:
        :return:
        """
        return len(self.buffer) >= (batch_size + 1)


    def store_experience(self, state, action, reward, next_state, done):
        """
        store experience to buffer
        :param experience:
        :return:
        """
        if len(self.buffer) >= self.max_capacity:
            del self.buffer[:1000]

        self.buffer.append((state.flatten(), action, reward, next_state.flatten(), done))


    def sample(self, **kwargs):
        """
            sample experience from buffer
        :param batch_size:
        :return:
        """
        # DRL
        if kwargs["name"] == "RL":
            batch_size = kwargs["batch_size"]
            experience = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*experience)
            state = np.array(state).reshape((batch_size, -1, self.nodes, self.nodes))
            next_state = np.array(next_state).reshape((batch_size, -1, self.nodes, self.nodes))
            return state, action, reward, next_state, done
        # GRU
        if kwargs["name"] == "RNN":
            states = []
            target_states = []
            idx = kwargs["idx"]
            batch_size = kwargs["batch_size"]
            experience = self.buffer[idx: idx + batch_size + 1]
            state, action, reward, next_state, done = zip(*experience)
            state = np.array(state)
            next_state = np.array(next_state)
            # --- get input target --- #
            input_state = state[:-1, :]
            target_state = state[-1, :]
            input_next_state = next_state[:-1, :]
            target_next_state = next_state[-1, :]
            # change input state's shape --> batch,4,144
            states.append(input_state.reshape(batch_size, -1, self.nodes * self.nodes))
            states.append(input_next_state.reshape(batch_size, -1, self.nodes * self.nodes))
            # change target state's shape -->1,4,144
            target_states.append(target_state.reshape(1, -1, self.nodes * self.nodes))
            target_states.append(target_next_state.reshape(1, -1, self.nodes * self.nodes))

            return states, target_states





