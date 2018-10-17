import numpy as np
import util_functions as uf
# from IPython.core.debugger import set_trace; set_trace()


class Memory():
	def __init__(self,n_inputs,n_outputs,**params):
		self._max_size = int(params.pop('max_size',1e4))
		self._loop_when_full = params.pop('loop_when_full',False)
		self._discrete_action = params.pop('discrete_action',False)
		self.n_inputs = [n_inputs] if not(isinstance(n_inputs,list)) else n_inputs
		self._dtype_obs,obs_dim = params.pop('dtype_obs',(np.float32,tuple(self.n_inputs)))
		
		self.n_outputs = n_outputs
		
		dt = np.dtype([('action', np.float32, self.n_outputs), ('obs', self._dtype_obs, obs_dim), ('next_obs', self._dtype_obs, obs_dim), ('reward', np.float32), ('done', np.bool_)])
		self._memory = np.empty(self._max_size, dtype=dt)
		
		self.reset()
		
	def reset(self):
		self._size = 0
		self._pos = -1
		
	def add_sample(self,action,obs,next_obs,reward,done):
		self._advance()
		self._memory[self._pos] = (action,obs,next_obs,reward,done)
		

	def is_full(self):
		return self._size == self._max_size

	def last_sample_done(self):
		return self._memory[self._pos]['done'] == True

	def _advance(self):
		if self._loop_when_full:
			self._pos = (self._pos + 1) % self._max_size
		else:
			if self.is_full():
				raise Warning('Trying to add too many entries to memory buffer, which is now full!')
			self._pos += 1
		
		if self._size < self._max_size:
			self._size += 1
			
	
	def _get_samples(self,inds =  None,multi_step =False):

		if multi_step:
			return self._get_multi_step_samples(inds,multi_step)

		elif inds is None:
			return dict(actions = self._memory['action'],
				   obs = [np.asarray(obs) for obs in self._memory['obs']],
				   next_obs = [np.asarray(obs) for obs in self._memory['next_obs']],
				   rewards = self._memory['reward'],
				   dones = self._memory['done'])
		else:
			entries = self._memory[inds]
			
			return dict(actions = entries['action'],
				   obs = [np.asarray(obs) for obs in entries['obs']],
				   next_obs = [np.asarray(obs) for obs in entries['next_obs']],
				   rewards = entries['reward'],
				   dones = entries['done'])



	def _get_multi_step_samples(self,inds,multi_step):
		to_get_inds = np.expand_dims(inds,1) + np.tile(np.arange(multi_step),(np.shape(inds)[0],1))
		entries = self._memory[to_get_inds]
		
		rewards = entries['reward'] * uf.mask_rewards_using_dones(entries['done'],axis=1) # Only keep rewards up to done
		dones = np.sum(entries['done'],1) # Finally, signal done if any within are done
		return dict(actions = entries[:,0]['action'],
				   obs = [np.asarray(obs) for obs in entries[:,0]['obs']],
				   next_obs = [np.asarray(obs) for obs in entries[:,-1]['next_obs']],
				   rewards = rewards,
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
	   