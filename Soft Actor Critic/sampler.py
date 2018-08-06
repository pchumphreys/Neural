import numpy as np

class Sampler():
    def __init__(self,policy,env,replaybuffer,log,max_episode_length = -1):
        self.policy = policy
        self.env = env
        self.replaybuffer = replaybuffer
        self.log = log
        
        self.max_episode_length = max_episode_length
        self.reset()
        
    def reset(self): 
        self.current_obs = False
        self.episodes = 0
        self.current_episode_reward = 0
        self.mean_episode_reward = 0

    def sample(self):
        if (self.current_obs is False):
            self.current_obs = self.env.reset()
            self.current_t = 0
            
        action = self.policy.get_action(self.current_obs)
        next_obs, reward, done, info = self.env.step(action)
        if self.policy.discrete == True:
            action = np.eye(self.policy.action_size)[action]
        self.replaybuffer.add_sample(action,self.current_obs,next_obs,reward,done)
        self.current_obs = next_obs
        
        self.current_episode_reward += reward

        if done or self.current_t == self.max_episode_length:
            self.current_obs = False   
            self.episodes += 1

            self.log.record('episode_reward',self.current_episode_reward)
            self.mean_episode_reward = (self.mean_episode_reward * (self.episodes - 1) + self.current_episode_reward) / self.episodes
            self.current_episode_reward = 0
            
    # Helper functions forwarded through rb.

    def batch_ready(self):
        return self.replaybuffer.batch_ready()

    def get_last_sample(self):
        return self.replaybuffer.get_last_sample()

    def get_samples(self):
        return self.replaybuffer.get_samples()

