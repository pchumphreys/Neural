import tensorflow as tf
import numpy as np

import util_functions as uf
from base_agent import Base_Agent
from value_functions import Qnet
from policies import Policy_Discrete_for_Qnet
from memories import Replay_Buffer
import atari_wrappers

class DQN_agent(Base_Agent):
	# This class handles the training of the networks
	def __init__(self,n_inputs,n_outputs,**params):

		super(DQN_agent,self).__init__(**params)
		
		self.n_inputs = list(n_inputs)
		self.n_outputs = n_outputs
		self._discrete_action=True

		self.lr = params['agent_params'].pop('lr',1e-3)
		self.discount = params['agent_params'].pop('discount',1e-3)
		self.tau = params['agent_params'].pop('tau',1e-3)

		self.soft_learning = params['agent_params'].pop('soft_learning',False)
		self.reward_scale = params['agent_params'].pop('reward_scale',1.0)
		self.double =  params['agent_params'].pop('double', False)
		self.huber_loss =  params['agent_params'].pop('huber_loss', True)
		self.clip_gradients =  params['agent_params'].pop('clip_gradients', False)
		self.train_steps_per_t = params['agent_params'].pop('train_steps_per_t',1)
		self.action_steps_per_train = params['agent_params'].pop('action_steps_per_train',1)
		self.steps_since_train = 1
		self.multi_step = params['agent_params'].pop('multi_step',False)
		if self.multi_step:
			self.discount = self.discount**self.multi_step

		assert not(self.soft_learning and self.double)
		
		if 'atari_env' in params and params['atari_env']:
			params['replay_buffer_params']['dtype_obs']=(atari_wrappers.LazyFrames,1)

		self._init_placeholders()
		
		self.qnet = Qnet(self.obs,self.n_outputs,params['network_spec'],scope='qnet')
		self.model_Q_params = self.qnet.get_params_internal()
		self.model_Q_outputs = self.qnet.outputs
		self.model_Q_predict_from_next_obs = tf.stop_gradient(tf.one_hot(tf.argmax(self.qnet.make_network(inputs = self.next_obs),axis=1),self.qnet.output_size))
		
		# Duplicate the Qnet with different variables for the target network
		self.tqnet = Qnet(self.next_obs,self.n_outputs,params['network_spec'],scope='tqnet')
		self.target_Q_outputs = self.tqnet.outputs 
		self.target_Q_params = self.tqnet.get_params_internal()  
		
		if self.soft_learning:
			# For soft learning
			# V = sum(p(s,a) * (q(s,a) - log(p(s,a)))
			#   = sum(exp(q)/z * (q - log(exp(q)/z)))
			#   = sum(p* (log(z)))
			#   = log(z)
			self.partition_function = tf.reduce_mean(self.target_Q_outputs,axis=1) + tf.log(tf.reduce_sum(tf.exp(self.target_Q_outputs - tf.reduce_mean(self.target_Q_outputs,axis=1,keepdims=True)),axis=1))
			self.target_V = self.partition_function

			params['policy_params']['action_choice'] = params['policy_params'].get('action_choice','Boltzmann')
			assert params['policy_params']['action_choice'] == 'Boltzmann' # Softmax on outputs


		self.policy = Policy_Discrete_for_Qnet(self.qnet,**params['policy_params'])

		self.rb = Replay_Buffer(self.n_inputs,self.n_outputs,multi_step = self.multi_step ,**params['replay_buffer_params'])
	
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
		
		self.train_ops = []
		self._init_qnet_training_op()
		
		self.target_Q_update = uf.update_target_network(self.model_Q_params,self.target_Q_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		self.train_ops.append(self.target_Q_update)


		self._finish_agent_setup()



	def _init_placeholders(self):

		self.actions = tf.placeholder(tf.float32,shape = [None,self.n_outputs],name = 'actions')
		self.raw_obs = tf.placeholder(tf.float32,shape = [None]+self.n_inputs,name = 'observations')
		self.raw_next_obs = tf.placeholder(tf.float32,shape = [None]+ self.n_inputs,name = 'next_observations')
		self.obs = self.raw_obs / 255
		self.next_obs = self.raw_next_obs /255
		self.rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
		self.dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')

		self.scaled_rewards = self.reward_scale * self.rewards
		
	def _construct_feed_dict(self,samples):
		return {self.actions : samples['actions'],
				self.raw_obs : samples['obs'],
				self.raw_next_obs : samples['next_obs'],
				self.dones : samples['dones'],
				self.rewards : samples['rewards']}

	def _init_qnet_training_op(self):
		

		with tf.variable_scope('Q_loss'):

			if self.double:
				target = tf.stop_gradient(self.scaled_rewards +  self.discount * (1-self.dones) * tf.reduce_sum(self.target_Q_outputs*self.model_Q_predict_from_next_obs,axis=1))
			elif self.soft_learning:
				target = tf.stop_gradient(self.scaled_rewards  +  self.discount * (1-self.dones) * self.target_V)
			else:
				target = tf.stop_gradient(self.scaled_rewards +  self.discount * (1-self.dones) * tf.reduce_max(self.target_Q_outputs,axis=1))
		
				
			if self.huber_loss:
				self.Q_Loss = tf.reduce_mean(tf.sqrt(1+tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - target))-1)
			else:
				self.Q_Loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - target))

			tf.summary.scalar('Q_loss', self.Q_Loss)
		
		train_op = self._get_regs_add_clip_make_optimizer(self.model_Q_params,self.Q_Loss,self.qnet._name)

		self.train_ops.append(train_op)
	
					

	def train(self,global_step):
		if self.rb.batch_ready():
			if self.steps_since_train == self.action_steps_per_train:
				for j in range(self.train_steps_per_t):
					samples = self.rb.get_random_batch()
					if self.multi_step:
						samples['rewards'] = uf.calc_discount(samples['rewards'],self.discount,axis=1)[:,0]
				
					losses = self._train(samples,global_step,[self.Q_Loss])
				self.steps_since_train = 1
			else:
				losses = False
				self.steps_since_train += 1


		else:
			losses = False
		return losses

	def add_sample(self,action,current_obs,next_obs,reward,done):

		if self._discrete_action == True:
			assert(not(isinstance(action,list)))
			action = np.eye(self.n_outputs)[action]
		else:
			assert(len(action)==self.n_outputs)

		self.rb.add_sample(action,current_obs,next_obs,reward,done)


			 