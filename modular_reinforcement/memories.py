import numpy as np
import util_functions as uf
# from IPython.core.debugger import set_trace; set_trace()


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
		self.obs = np.zeros([self._max_size,self.n_inputs])
		self.next_obs = np.zeros([self._max_size,self.n_inputs])
		self.rewards = np.zeros(self._max_size)
		self.dones = np.zeros(self._max_size)
		
	def add_sample(self,action,obs,next_obs,reward,done):
		self._advance()

		if self._discrete_action == True:
			assert(not(isinstance(action,list)))
			action = np.eye(self.n_outputs)[action]
		else:
			assert(len(action)==self.n_outputs)

		self.actions[self._pos] = action
		self.obs[self._pos] = obs
		self.next_obs[self._pos] = next_obs
		self.rewards[self._pos] = reward
		self.dones[self._pos] = done
		

	def is_full(self):
		return self._size == self._max_size

	def last_sample_done(self):
		return self.dones[self._pos] == True

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

			return dict(actions = self.actions[:self._size],
				   obs = self.obs[:self._size],
				   next_obs = self.next_obs[:self._size],
				   rewards = self.rewards[:self._size],
				   dones = self.dones[:self._size])
		else:
			
			return dict(actions = self.actions[inds],
					   obs = self.obs[inds],
					   next_obs = self.next_obs[inds],
					   rewards = self.rewards[inds],
					   dones = self.dones[inds])

	def _get_samples(self,inds =  None,multi_step =False):

		if multi_step:
			return self._get_multi_step_samples(inds,multi_step)

		elif inds is None:

			return dict(actions = self.actions[:self._size],
				   obs = self.obs[:self._size],
				   next_obs = self.next_obs[:self._size],
				   rewards = self.rewards[:self._size],
				   dones = self.dones[:self._size])
		else:
			
			return dict(actions = self.actions[inds],
					   obs = self.obs[inds],
					   next_obs = self.next_obs[inds],
					   rewards = self.rewards[inds],
					   dones = self.dones[inds])

	def _get_multi_step_samples(self,inds,multi_step):
		to_get_inds = np.expand_dims(inds,1) + np.tile(np.arange(multi_step),(np.shape(inds)[0],1))
		final_step_inds = to_get_inds[:,-1]
		dones = self.dones[to_get_inds]
		rewards = self.rewards[to_get_inds] * uf.mask_rewards_using_dones(dones,axis=1) # Only keep rewards up to done
		dones = np.sum(self.dones[to_get_inds],1) # Finally, signal done if any within are done
		return dict(actions = self.actions[inds],
			   obs = self.obs[inds],
			   next_obs = self.next_obs[final_step_inds],
			   rewards = rewards, # Note that rewards is still large, since need to apply discount appropriately.
			   dones = dones)

	def get_last_sample(self):
		ind = [self._pos]
		return self._get_samples(ind)

	def get_all_samples(self):
		return self._get_samples()


class Replay_Buffer(Memory):
	def __init__(self,n_inputs,n_outputs,**params):
		self._min_pool_size = int(params.pop('min_pool_size',1000))
		self._batch_size = int(params.pop('batch_size',32))
		self._multi_step = params.pop('multi_step',False)

		params['loop_when_full'] = params.get('loop_when_full',True)
		assert params['loop_when_full'] == True

		super(Replay_Buffer,self).__init__(n_inputs,n_outputs,**params)

	def get_random_batch(self,batch_size = None):
		if batch_size is None:
			batch_size = self._batch_size
		size_adjust = (self._multi_step-1) if self._multi_step else 0
		inds = np.random.randint(0,self._size-size_adjust,batch_size)
		return self._get_samples(inds,self._multi_step)

	def batch_ready(self):
		return self._size >= self._min_pool_size
	   