import numpy as np

class Runner():
    def __init__(self,env,agent,**params):
        self.env = env
        self.agent = agent

        self.episode_finished_callback = params.pop('episode_finished_callback',None)
        self.max_episodes = params.pop('max_episodes',1)
        self.max_episode_length = params.pop('max_episodes',1)
        self.verbose = params.pop('verbose',True)
        self.reset()


        
    def reset(self): 
        self.done = True
        self.episodes = 0
        self.episode_rewards = []
        self.episode_average_losses = []
    
    def run(self):
        self.reset()
        
        for i in range(self.max_episodes):
            
            while True:
                self.sample()
                losses = self.agent.train()
                if not(losses is False):
                    self.current_episode_losses.append(losses)   

                if self.done or self.current_t == self.max_episode_length:
                    self.done = True   
                    self.episodes += 1

                    self.episode_rewards.append(self.current_episode_reward)
                    if len(self.current_episode_losses):
                        self.episode_average_losses.append(np.mean(self.current_episode_losses))

                    if not(self.episode_finished_callback is None):
                        self.episode_finished_callback(self)

                    self.agent.episode_finished()

                    break
        


    def sample(self):
        if self.done:
            self.current_obs = self.env.reset()
            self.current_t = 0
            self.current_episode_reward = 0
            self.current_episode_losses = []
            
        action = self.agent.get_action(self.current_obs)
        next_obs, reward, self.done, info = self.env.step(action)

        self.agent.add_sample(action,self.current_obs,next_obs,reward,self.done)
        self.current_obs = next_obs
        self.current_episode_reward += reward
        

