import numpy as np

class ReplayBuffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.is_done = []

        self.index_to_remove = 0

    @property
    def is_full(self):
        return len(self.rewards) >= self.replay_buffer_size

    def add_data(self, state, action, next_state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.is_done.append(done)

        if self.is_full:
            self.actions = np.array(self.actions)
            self.states = np.array(self.states)
            self.next_states = np.array(self.next_states)
            self.rewards = np.array(self.rewards)
            self.is_done = np.array(self.is_done)

    def update_data(self, state, action, next_state, reward, done):
        self.states[self.index_to_remove] = state
        self.actions[self.index_to_remove] = action
        self.next_states[self.index_to_remove] = next_state
        self.rewards[self.index_to_remove] = reward
        self.is_done[self.index_to_remove] = done
        self.index_to_remove = (self.index_to_remove + 1) % self.replay_buffer_size

    def sample(self, sample_size):
        ### sample_states, sample_next_states: (sample_size x feature_dims)
        ### sample_actions, sample_rewards, sample_is_done: (sample_size x ,)

        sample_indices = np.random.choice(self.replay_buffer_size, sample_size)
        sample_states = self.states[sample_indices]
        sample_actions = self.actions[sample_indices]
        sample_next_states = self.next_states[sample_indices]
        sample_rewards = self.rewards[sample_indices]
        sample_is_done = self.is_done[sample_indices]
        # print(sample_actions.shape)
        return sample_states, sample_actions, sample_next_states, sample_rewards, sample_is_done

