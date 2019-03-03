class Simulation(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
        #logging
        self.rewards = []
    def play(self, num_episodes, render):
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                action = self.agent.select_action(state)
                state, reward, done, info = self.env.step(action)
                total_reward+=reward
            self.rewards.append(total_reward)