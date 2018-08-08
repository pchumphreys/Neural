import numpy as np
import tensorflow as tf
import os


class Runner():
    def __init__(self,env,agent,**params):
        self.env = env
        self.agent = agent

        self.saver = params.pop('saver',None)
        self.episode_finished_callback = params.pop('episode_finished_callback',None)

        self.max_episodes = params['runner'].pop('max_episodes',1)
        self.max_episode_length = params['runner'].pop('max_episode_length',-1)
        
        self.checkpoint_interval = params['runner'].pop('checkpoint_interval',100)
        self.log_dir = params.get('log_dir',None)

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

                    if (self.episodes % self.checkpoint_interval == 0) and not(self.saver is None):
                        self.saver.save(tf.get_default_session(),os.path.join(self.log_dir,'model'),global_step = self.episodes)


                    break

            if not(self.saver is None):
                self.saver.save(tf.get_default_session(),os.path.join(self.log_dir,'final_model'),global_step = self.episodes)

            if not(self.log_dir is None):
                np.savetxt(os.path.join(log_dir,'rewards.csv'),self.episode_rewards)
                
        


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
        self.current_t += 1
        

