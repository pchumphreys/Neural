import numpy as np

class Replay_Buffer():
	def __init__(self,n_inputs,n_outputs,discrete_action= True,**params):
		self._max_size = int(params.pop('max_buffer_size',1e4))
		self._min_pool_size = int(params.pop('min_pool_size',1000))
		self._batch_size = int(params.pop('batch_size',32))
		self._discrete_action = discrete_action
		self.n_outputs = n_outputs
		self.n_inputs = n_inputs
		self.reset()
		
	def reset(self):
		self._size = 0
		self._pos = 0
		
		self.actions = np.zeros([self._max_size,self.n_outputs])
		self.observations = np.zeros([self._max_size,self.n_inputs])
		self.next_observations = np.zeros([self._max_size,self.n_inputs])
		self.rewards = np.zeros(self._max_size)
		self.dones = np.zeros(self._max_size)
		
	def add_sample(self,action,obs,next_obs,reward,done):
        if self._discrete_action == True:
            action = np.eye(self.policy.action_size)[action]

		self.actions[self._pos] = action
		self.observations[self._pos] = obs
		self.next_observations[self._pos] = next_obs
		self.rewards[self._pos] = reward
		self.dones[self._pos] = done
		
		self._advance()
	
	def _advance(self):
		self._pos = (self._pos + 1) % self._max_size
		
		if self._size < self._max_size:
			self._size += 1
			
	def get_samples(self):
		inds = np.random.randint(0,self._size,self._batch_size)
		return dict(actions = self.actions[inds],
				   observations = self.observations[inds],
				   next_observations = self.next_observations[inds],
				   rewards = self.rewards[inds],
				   dones = self.dones[inds])

	
	def get_last_sample(self):
		last_pos = [(self._pos-1) % self._max_size]
		return dict(actions = self.actions[last_pos],
				   observations = self.observations[last_pos],
				   next_observations = self.next_observations[last_pos],
				   rewards = self.rewards[last_pos],
				   dones = self.dones[last_pos])
	
	def batch_ready(self):
		return self._size >= self._min_pool_size
	   