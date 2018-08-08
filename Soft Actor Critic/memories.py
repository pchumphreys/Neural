import numpy as np


class Memory():
	def __init__(self,n_inputs,n_outputs,**params):
		self._max_size = int(params.pop('max_size',1e4))
		self._loop_when_full = params.pop('loop_when_full',False)
		self._discrete_action = params.pop('discrete_action',False)
		self.n_outputs = n_outputs
		self.n_inputs = n_inputs
		
		self.reset()
		
	def reset(self):
		self._size = 0
		self._pos = -1
		
		self.actions = np.zeros([self._max_size,self.n_outputs])
		self.observations = np.zeros([self._max_size,self.n_inputs])
		self.next_observations = np.zeros([self._max_size,self.n_inputs])
		self.rewards = np.zeros(self._max_size)
		self.dones = np.zeros(self._max_size)
		
	def add_sample(self,action,obs,next_obs,reward,done):
		self._advance()

		if self._discrete_action == True:
			assert(len(action)==1)
			action = np.eye(self.n_outputs)[action]
		else:
			assert(len(action)==self.n_outputs)

		self.actions[self._pos] = action
		self.observations[self._pos] = obs
		self.next_observations[self._pos] = next_obs
		self.rewards[self._pos] = reward
		self.dones[self._pos] = done
		

	def is_full(self):
		return self._size == self._max_size

	def last_sample_done(self):
		return self.dones[self._pos] = True

	def _advance(self):
		if self._loop_when_full:
			self._pos = (self._pos + 1) % self._max_size
		else:
			if self.is_full():
				raise Warning('Trying to add too many entries to memory buffer, which is now full!')
			self._pos += 1
		
		if self._size < self._max_size:
			self._size += 1
			
	def _get_samples(self,inds =  None):
		if inds is None:

			return dict(actions = self.actions,
				   observations = self.observations,
				   next_observations = self.next_observations,
				   rewards = self.rewards,
				   dones = self.dones)
		else:
			
			return dict(actions = self.actions[inds],
					   observations = self.observations[inds],
					   next_observations = self.next_observations[inds],
					   rewards = self.rewards[inds],
					   dones = self.dones[inds])

	def get_last_sample(self):
		ind = [self._pos]
		return self._get_samples(ind)

	def get_all_samples(self):
		return self._get_samples()


class Replay_Buffer(Memory):
	def __init__(self,**params):
		self._min_pool_size = int(params.pop('min_pool_size',1000))
		self._batch_size = int(params.pop('batch_size',32))
		params['loop_when_full'] = params.get('loop_when_full',True)
		assert params['loop_when_full'] == True

		super(Replay_Buffer,self).__init__(**params)

	def get_random_batch(self):
		inds = np.random.randint(0,self._size,self._batch_size)
		return self._get_samples(inds)

	def batch_ready(self):
		return self._size >= self._min_pool_size
	   