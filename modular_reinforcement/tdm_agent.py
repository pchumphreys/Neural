import tensorflow as tf
import numpy as np

import util_functions as uf
from base_agent import Base_Agent
from value_functions import Qnet
from policies import Policy_Discrete_for_Qnet
from memories import Replay_Buffer

class TDM_agent(Base_Agent):
	# This class handles the training of the networks
	def __init__(self,n_inputs,n_outputs,**params):

		super(TDM_agent,self).__init__(**params)
		
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs

		self.lr = params['agent_params'].pop('lr',1e-3)
		self.discount = params['agent_params'].pop('discount',1e-3)
		self.tau = params['agent_params'].pop('tau',1e-3)
		self.max_td = params['agent_params'].pop('max_td',3)

		self.soft_learning = params['agent_params'].pop('soft_learning',False)
		self.reward_scale = params['agent_params'].pop('reward_scale',1.0)
		self.double =  params['agent_params'].pop('double', False)
		self.reward_scale = params['agent_params'].pop('reward_scale',1.0)
		self.huber_loss =  params['agent_params'].pop('huber_loss', True)
		self.clip_gradients =  params['agent_params'].pop('clip_gradients', False)
		self.train_steps_per_t = params['agent_params'].pop('train_steps_per_t',1)
		self.q_train_steps_per_t = params['agent_params'].pop('q_train_steps_per_t',1)
		self.extra_q_train_steps_per_t = self.q_train_steps_per_t - self.train_steps_per_t
		assert self.extra_q_train_steps_per_t >= 0

		self.multi_step = params['agent_params'].pop('multi_step',False)
		if self.multi_step:
			self.discount = self.discount**self.multi_step
			
		
		assert not(self.soft_learning and self.double)

		self._init_placeholders()

		### QNET
		self.qnet = Qnet(self.obs,self.n_outputs,params['network_spec'],scope='qnet')
		self.model_Q_params = self.qnet.get_params_internal()
		self.model_Q_outputs = self.qnet.outputs
		
		### FNET
		self.fnet = Qnet([self.obs,self.actions,self.one_hot_tds],self.n_inputs,params['network_spec'],scope='fnet')
		self.model_F_params = self.fnet.get_params_internal()
		self.model_F_outputs = self.fnet.outputs
		
		### RNET
		self.rnet = Qnet([self.obs,self.actions,self.one_hot_tds],1,params['network_spec'],scope='rnet')
		self.model_R_params = self.rnet.get_params_internal()
		self.model_R_outputs = self.rnet.outputs

		### ENET
		if self.soft_learning:
			self.enet = Qnet([self.obs,self.actions,self.one_hot_tds],1,params['network_spec'],scope='enet')
			self.model_E_params = self.enet.get_params_internal()
			self.model_E_outputs = self.enet.outputs

			self.model_Q_predict_action_from_next_obs = tf.stop_gradient(tf.one_hot(tf.multinomial(self.qnet.make_network(inputs = self.next_obs),1)[:,0],self.qnet.output_size))
		else:
			self.model_Q_predict_action_from_next_obs = tf.stop_gradient(tf.one_hot(tf.argmax(self.qnet.make_network(inputs = self.next_obs),axis=1),self.qnet.output_size))
		
		# Duplicate the Fnet with different variables for the target network
		self.tfnet = Qnet([self.next_obs,self.model_Q_predict_action_from_next_obs,self.one_hot_next_tds],self.n_inputs,params['network_spec'],scope='tfnet')
		self.target_F_outputs = self.tfnet.outputs 
		self.target_F_params = self.tfnet.get_params_internal()
		self.target_F_from_obs = self.tfnet.make_network(inputs = [self.obs,self.actions,self.one_hot_tds])  

		# Duplicate the Rnet with different variables for the target network
		self.trnet = Qnet([self.next_obs,self.model_Q_predict_action_from_next_obs,self.one_hot_next_tds],1,params['network_spec'],scope='trnet')
		self.target_R_outputs = self.trnet.outputs 
		self.target_R_params = self.trnet.get_params_internal()
		self.target_R_from_obs = self.trnet.make_network(inputs = [self.obs,self.actions,self.one_hot_tds])
		  
		# Duplicate the Qnet with different variables for the target network
		self.tqnet = Qnet(tf.add(self.next_obs,self.td_is_not_zero * self.target_F_outputs),self.n_outputs,params['network_spec'],scope='tqnet')
		self.target_Q_outputs = self.tqnet.outputs 
		self.target_Q_params = self.tqnet.get_params_internal()  
		
		# Duplicate the Enet with different variables for the target network
		if self.soft_learning:
			self.tenet = Qnet([self.next_obs,self.model_Q_predict_action_from_next_obs,self.one_hot_next_tds],1,params['network_spec'],scope='tenet')
			self.target_E_outputs = self.tenet.outputs 
			self.target_E_params = self.tenet.get_params_internal()
			self.target_E_from_obs = self.tenet.make_network(inputs = [self.obs,self.actions,self.one_hot_tds])

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

		self.rb = Replay_Buffer(self.n_inputs,self.n_outputs,discrete_action=True,multi_step = self.multi_step,**params['replay_buffer_params'])
	
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
		
		self.train_ops = []
		self._init_training_ops()
		
		self.target_Q_update = uf.update_target_network(self.model_Q_params,self.target_Q_params,tau=self.tau,update_op_control_dependencies=self.q_train_op)
		self.target_R_update = uf.update_target_network(self.model_R_params,self.target_R_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		self.target_F_update = uf.update_target_network(self.model_F_params,self.target_F_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		if self.soft_learning:
			self.target_E_update = uf.update_target_network(self.model_E_params,self.target_E_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		
		self.train_ops.append([self.target_R_update,self.target_F_update])
		if self.soft_learning:
			self.target_E_update

		self.q_train_ops = tf.group(self.q_train_op,self.target_Q_update)
		self.train_ops = tf.group(self.train_ops,self.q_train_ops)
		
		self.loss_ops = [self.R_Loss,self.F_Loss,self.Q_Loss]
		if self.soft_learning:
			self.loss_ops.append(self.E_Loss)

		self._finish_agent_setup()

	def _init_placeholders(self):
		
		self.actions = tf.placeholder(tf.float32,shape = [None,self.n_outputs],name = 'actions')
		self.obs = tf.placeholder(tf.float32,shape = [None,self.n_inputs],name = 'observations')
		self.next_obs = tf.placeholder(tf.float32,shape = [None,self.n_inputs],name = 'next_observations')
		self.rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
		self.dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')
		self.tds = tf.placeholder(tf.int32,shape = [None],name = 'tds')

		self.scaled_rewards = self.reward_scale * self.rewards

		self.one_hot_tds = tf.one_hot(self.tds,depth = self.max_td)
		self.next_tds = self.tds - 1
		self.one_hot_next_tds = tf.one_hot(self.next_tds,depth = self.max_td)
		self.td_is_not_zero = tf.expand_dims(tf.to_float(tf.greater(self.tds,0)),1)

	def _construct_feed_dict(self,samples):
		return {self.actions : samples['actions'],
				self.obs : samples['obs'],
				self.next_obs : samples['next_obs'],
				self.dones : samples['dones'],
				self.rewards : samples['rewards'],
				self.tds : samples['tds']}

	def _init_training_ops(self):
		

		with tf.variable_scope('Q_loss'):
			if self.double:
				target = tf.stop_gradient(self.scaled_rewards + self.td_is_not_zero * self.discount * ((1-self.dones) * self.target_R_outputs) +  tf.pow(self.discount,tf.cast(self.tds+1,tf.float32)) *  (1-self.dones) * tf.reduce_sum(self.target_Q_outputs*self.model_Q_predict_action_from_next_obs ,axis=1))	
			elif self.soft_learning:
				target = tf.stop_gradient(self.scaled_rewards + self.td_is_not_zero * self.discount * (1-self.dones) * (self.target_R_outputs + self.target_E_outputs) +  tf.pow(self.discount,tf.cast(self.tds+1,tf.float32)) *  (1-self.dones) * self.target_V)
			else:
				target = tf.stop_gradient(self.scaled_rewards + self.td_is_not_zero * self.discount * (1-self.dones) * self.target_R_outputs +  tf.pow(self.discount,tf.cast(self.tds+1,tf.float32)) *  (1-self.dones) * tf.reduce_max(self.target_Q_outputs ,axis=1))
				
			if self.huber_loss:
				self.Q_Loss = tf.reduce_mean(tf.sqrt(1+tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - target))-1)
			else:
				self.Q_Loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - target))

			tf.summary.scalar('Q_loss', self.Q_Loss)
		
		self.q_train_op = self._get_regs_add_clip_make_optimizer(self.model_Q_params,self.Q_Loss,self.qnet._name)


		with tf.variable_scope('R_loss'):

			target = tf.stop_gradient(self.scaled_rewards +   self.discount * (1-self.dones) * self.td_is_not_zero * self.target_R_outputs)
			self.R_Loss = tf.reduce_mean(tf.square(self.model_R_outputs - target))

			tf.summary.scalar('R_loss', self.R_Loss)
		
		train_op = self._get_regs_add_clip_make_optimizer(self.model_R_params,self.R_Loss,self.rnet._name)
		self.train_ops.append(train_op)

		with tf.variable_scope('F_loss'):

			target = tf.stop_gradient((self.next_obs - self.obs) +  self.td_is_not_zero * self.target_F_outputs)
			self.F_Loss = tf.reduce_mean(tf.square(self.model_F_outputs - target))

			tf.summary.scalar('R_loss', self.F_Loss)
		
		train_op = self._get_regs_add_clip_make_optimizer(self.model_F_params,self.F_Loss,self.fnet._name)
		self.train_ops.append(train_op)

		if self.soft_learning:
			with tf.variable_scope('E_loss'):

				log_responsible_policy_output = tf.reduce_sum((self.policy.log_policy_outputs)* self.actions,axis=1)
				target = tf.stop_gradient(-log_responsible_policy_output +   self.discount * (1-self.dones) * self.td_is_not_zero * self.target_E_outputs)
				self.E_Loss = tf.reduce_mean(tf.square(self.model_E_outputs - target))

				tf.summary.scalar('E_loss', self.E_Loss)
			
			train_op = self._get_regs_add_clip_make_optimizer(self.model_E_params,self.E_Loss,self.enet._name)
			self.train_ops.append(train_op)
	
					

	def train(self):
		if self.rb.batch_ready():
			for j in range(self.train_steps_per_t):
				samples = self.rb.get_random_batch()
				if self.multi_step:
					samples['rewards'] = uf.calc_discount(samples['rewards'],self.discount,axis=1)[:,0]
				
				samples['tds'] = np.random.randint(0,self.max_td,size = self.rb._batch_size)
				losses = self._train(samples,self.loss_ops)

			if self.extra_q_train_steps_per_t:
				for j in range(self.extra_q_train_steps_per_t):
					samples = self.rb.get_random_batch()

					samples['tds'] = np.random.randint(0,self.max_td,size = self.rb._batch_size)

					feed_dict = self._construct_feed_dict(samples)  
					tf.get_default_session().run([self.q_train_ops], feed_dict = feed_dict)
			
		else:
			losses = False
		return losses

	def add_sample(self,action,current_obs,next_obs,reward,done):
		self.rb.add_sample(action,current_obs,next_obs,reward,done)


			 