import tensorflow as tf
import numpy as np

import util_functions as uf

from base_agent import Base_Agent
from value_functions import Qnet
from policies import Policy_Discrete
from memories import Memory

class A2C_agent(Base_Agent):
	# This class handles the training of the networks
	def __init__(self,n_inputs,n_outputs,**params):

		super(A2C_agent,self).__init__(**params)
		
		self.n_inputs = list(n_inputs)
		self.n_outputs = n_outputs

		self.lr = params['agent_params'].pop('lr',1e-3)
		self.discount = params['agent_params'].pop('discount',0.99)
		self.epsilon = params['agent_params'].pop('epsilon',1e-5)
		self.entropy_weight = params['agent_params'].pop('entropy_weight',0.01)
		self.vf_weight = params['agent_params'].pop('vf_weight',0.5)
		
		self.clip_gradients =  params['agent_params'].pop('clip_gradients', False)
		self.t_steps_per_train = params['agent_params'].pop('t_steps_per_train',50)
		
		self._init_placeholders()

		self.model_v = Qnet(self.obs,1,params['network_spec'])
		self.model_v_params = self.model_v.get_params_internal()
		self.model_v_outputs = self.model_v.outputs
		
		if 'policy_params' in params:
			params['policy_params']['action_choice'] = params['policy_params'].get('action_choice','Boltzmann')
		else:
			params['policy_params'] = {'action_choice' : 'Boltzmann'}
		assert params['policy_params']['action_choice'] == 'Boltzmann' # Softmax on outputs

		self.policy = Policy_Discrete(self.obs,self.n_outputs,params['network_spec'],**params['policy_params'])
		self.policy_params = self.policy.get_params_internal()

		self.memory = Memory(self.n_inputs,self.n_outputs,discrete_action=True,max_size = self.t_steps_per_train)
	
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr,epsilon=self.epsilon)
		
		self.train_ops = []
		self._init_training_op()
		
		self._finish_agent_setup()
		
	def _init_placeholders(self):

		self.actions = tf.placeholder(tf.float32,shape = [None,self.n_outputs],name = 'actions')
		self.obs = tf.placeholder(tf.float32,shape = [None]+self.n_inputs,name = 'observations')
		self.val_targets = tf.placeholder(tf.float32,shape = [None], name = 'value_targets')
		self.advs = tf.placeholder(tf.float32,shape = [None], name = 'advantages')
			  
	def _construct_feed_dict(self,samples):
		return {self.actions : samples['actions'],
				self.obs : samples['obs'],
				self.val_targets : samples['val_targets'],
				self.advs : samples['advs']}
   

	def _init_training_op(self):
		
		with tf.variable_scope('policy_loss'):
			self.policy_output_for_action = tf.reduce_sum(self.policy.policy_outputs * self.actions,axis=1)
			self.raw_policy_loss = -tf.reduce_mean(tf.log(self.policy_output_for_action)*self.advs)
			self.policy_entropy = -tf.reduce_mean(self.policy.policy_outputs * self.policy.log_policy_outputs)

			self.policy_loss = self.raw_policy_loss - self.entropy_weight * self.policy_entropy
			tf.summary.scalar('policy_loss', self.policy_loss)
		
		with tf.variable_scope('val_loss'):
			self.val_loss = tf.reduce_mean(tf.square(self.model_v_outputs - self.val_targets))
			tf.summary.scalar('val_loss', self.val_loss)

		train_op = self._get_regs_add_clip_make_optimizer(self.model_v_params + self.policy_params,self.vf_weight*self.val_loss + self.policy_loss)

		self.train_ops.append(train_op)

		
	def train(self,global_step):

		if (self.memory.last_sample_done() or self.memory.is_full()):
			samples = self.memory.get_all_samples()

			if self.memory.is_full():
				values = self.get_values(np.append(samples['obs'],[np.array(samples['next_obs'])[-1,:]],axis=0))
				bootstrap = values[-1] # If episode isnt done, need to add on val. Should make so that can run indep of whether episode done
			else:
				values = np.append(self.get_values(samples['obs']),0)
				bootstrap = 0

			samples['val_targets']  = uf.calc_discount(np.append(samples['rewards'],bootstrap),self.discount)[:-1] #discounted_total_rewards
			# Generalised advantage estimation
			samples['advs'] = uf.calc_discount(samples['rewards'] + self.discount * values[1:] - values[:-1],self.discount)

			losses = self._train(samples,global_step,[self.val_loss,self.policy_loss])

			self.memory.reset()

		else:
			losses = False
		return losses

	def get_values(self,obs):
		return np.squeeze(self.model_v_outputs.eval(feed_dict = {self.obs : obs}))

	def add_sample(self,action,current_obs,next_obs,reward,done):
		self.memory.add_sample(action,current_obs,next_obs,reward,done)
