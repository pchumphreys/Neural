import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import time

class Runner():
	def __init__(self,env,agent,**params):
		self.env = env
		self.agent = agent

		self.episode_finished_callback = params.pop('episode_finished_callback',None)
		
		self.max_episodes = params['runner_params'].pop('max_episodes',1)
		self.max_episode_length = params['runner_params'].pop('max_episode_length',-1)
		
		self.log_dir = params.get('log_dir',None)
		self.checkpoint_interval = params['runner_params'].pop('checkpoint_interval',100)
		if not(self.log_dir is None):
			self.saver = tf.train.Saver(max_to_keep=1)
		self.reset()
			
		
	def reset(self): 
		self.done = True
		self.episodes = 0
		self.global_t = 0
		self.global_trains = 0
		self.avg_sample_time = 0
		self.avg_train_time = 0 
		self.episode_rewards = []
		self.episode_average_losses = []
		
	def run(self):
		self.reset()
		
		try:
			for i in range(self.max_episodes):
				
				while True:
					start_time = time.time()
					self.sample()
					self.avg_sample_time = (self.avg_sample_time * self.global_t + time.time() - start_time)/(self.global_t + 1)
					
					start_time = time.time()
					losses = self.agent.train()

					if not(losses is False):

						self.avg_train_time = (self.avg_train_time * self.global_trains + time.time() - start_time)/(self.global_trains + 1)						
						self.current_episode_losses.append(losses)
						self.global_trains += 1 

					if (self.done or self.current_t == self.max_episode_length):
						break

					self.global_t += 1
		

				self.done = True   
				self.episodes += 1

				self.episode_rewards.append(self.current_episode_reward)
				if len(self.current_episode_losses):
					self.episode_average_losses.append(np.mean(self.current_episode_losses))

				if not(self.episode_finished_callback is None):
					self.episode_finished_callback(self)

				self.agent.episode_finished()

				if (self.episodes % self.checkpoint_interval == 0) and not(self.log_dir is None):
					self.saver.save(tf.get_default_session(),os.path.join(self.log_dir,'model'),global_step = self.episodes)
			

			if not(self.log_dir is None):
				self.saver.save(tf.get_default_session(),os.path.join(self.log_dir,'final_model'))
				np.savetxt(os.path.join(self.log_dir,'rewards.csv'),self.episode_rewards)
					
		except KeyboardInterrupt:
			print('Keyboard interupt detected')

		self.env.close()

	def test(self,render=True,optimal_action = True):

		self.reset()
		
		while True:
			self.sample(optimal_action = optimal_action, train = False)
			if render:
				self.env.render()
			if self.done or (self.current_t == self.max_episode_length):
				break
		print('Episode reward is %d' % (self.current_episode_reward))

		self.env.close()


	def sample(self, optimal_action = False, train = True):
		if self.done:
			self.current_obs = self.env.reset()
			self.current_t = 0
			self.current_episode_reward = 0
			self.current_episode_losses = []
			
		action = self.agent.get_action(self.current_obs, optimal_action = optimal_action)
		next_obs, reward, self.done, info = self.env.step(action)

		if train:   
			self.agent.add_sample(action,self.current_obs,next_obs,reward,self.done)
		self.current_obs = next_obs
		self.current_episode_reward += reward
		self.current_t += 1
		

