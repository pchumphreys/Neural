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

		self.reward_scale = params['agent_params'].pop('reward_scale',1.0)
		self.huber_loss =  params['agent_params'].pop('huber_loss', True)
		self.clip_gradients =  params['agent_params'].pop('clip_gradients', False)
		self.train_steps_per_t = params['agent_params'].pop('train_steps_per_t',1)
		
		self._init_placeholders()

		### QNET
		self.qnet = Qnet(self.obs,self.n_outputs,params['network_spec'],scope='qnet')
		self.model_Q_params = self.qnet.get_params_internal()
		self.model_Q_outputs = self.qnet.outputs
		self.model_Q_predict_action_from_next_obs = tf.stop_gradient(tf.one_hot(tf.argmax(self.qnet.make_network(inputs = self.next_obs),axis=1),self.qnet.output_size))
		
		### FNET
		self.fnet = Qnet([self.obs,self.actions,self.one_hot_tds],self.n_inputs,params['network_spec'],scope='fnet')
		self.model_F_params = self.fnet.get_params_internal()
		self.model_F_outputs = self.fnet.outputs
		
		### RNET
		self.rnet = Qnet([self.obs,self.actions,self.one_hot_tds],1,params['network_spec'],scope='rnet')
		self.model_R_params = self.rnet.get_params_internal()
		self.model_R_outputs = self.rnet.outputs

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
		self.tqnet = Qnet(tf.add(self.obs,self.target_F_from_obs),self.n_outputs,params['network_spec'],scope='tqnet')
		self.target_Q_outputs = self.tqnet.outputs 
		self.target_Q_params = self.tqnet.get_params_internal()  
		
		
		
		
		self.policy = Policy_Discrete_for_Qnet(self.qnet,**params['policy_params'])

		self.rb = Replay_Buffer(self.n_inputs,self.n_outputs,discrete_action=True,**params['replay_buffer_params'])
	
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
		
		self.train_ops = []
		self._init_training_ops()
		
		self.target_Q_update = uf.update_target_network(self.model_Q_params,self.target_Q_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		self.target_R_update = uf.update_target_network(self.model_R_params,self.target_R_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		self.target_F_update = uf.update_target_network(self.model_F_params,self.target_F_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		
		self.train_ops.append([self.target_Q_update,self.target_R_update,self.target_F_update])


		self._finish_agent_setup()

	def _init_placeholders(self):
		
		self.actions = tf.placeholder(tf.float32,shape = [None,self.n_outputs],name = 'actions')
		self.obs = tf.placeholder(tf.float32,shape = [None,self.n_inputs],name = 'observations')
		self.next_obs = tf.placeholder(tf.float32,shape = [None,self.n_inputs],name = 'next_observations')
		self.rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
		self.dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')
		self.tds = tf.placeholder(tf.int32,shape = [None],name = 'tds')
		self.one_hot_tds = tf.one_hot(self.tds,depth = self.max_td)
		self.next_tds = self.tds - 1
		self.one_hot_next_tds = tf.one_hot(self.next_tds,depth = self.max_td)
		self.td_is_zero = tf.expand_dims(tf.to_float(tf.equal(self.tds,0)),1)

	def _construct_feed_dict(self,samples):
		return {self.actions : samples['actions'],
				self.obs : samples['obs'],
				self.next_obs : samples['next_obs'],
				self.dones : samples['dones'],
				self.rewards : samples['rewards'],
				self.tds : samples['tds']}

	def _init_training_ops(self):
		

		with tf.variable_scope('Q_loss'):

			target = tf.stop_gradient(self.target_R_from_obs +  tf.pow(self.discount,tf.cast(self.tds+1,tf.float32)) * tf.reduce_max(self.target_Q_outputs,axis=1))
				
			if self.huber_loss:
				self.Q_Loss = tf.reduce_mean(tf.sqrt(1+tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - target))-1)
			else:
				self.Q_Loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - target))

			tf.summary.scalar('Q_loss', self.Q_Loss)
		
		train_op = self._get_regs_add_clip_make_optimizer(self.model_Q_params,self.Q_Loss,self.qnet._name)

		self.train_ops.append(train_op)

		with tf.variable_scope('R_loss'):

			target = tf.stop_gradient(self.rewards +   self.discount * (1-self.dones) * self.td_is_zero * self.target_R_outputs)
			self.R_Loss = tf.reduce_mean(tf.square(self.model_R_outputs - target))

			tf.summary.scalar('R_loss', self.R_Loss)
		
		train_op = self._get_regs_add_clip_make_optimizer(self.model_R_params,self.R_Loss,self.rnet._name)

		self.train_ops.append(train_op)

		with tf.variable_scope('F_loss'):

			target = tf.stop_gradient((self.next_obs - self.obs) +  self.td_is_zero * self.target_F_outputs)
			self.F_Loss = tf.reduce_mean(tf.square(self.model_F_outputs - target))

			tf.summary.scalar('R_loss', self.F_Loss)
		
		train_op = self._get_regs_add_clip_make_optimizer(self.model_F_params,self.F_Loss,self.fnet._name)

		self.train_ops.append(train_op)
	
					

	def train(self):
		if self.rb.batch_ready():
			for j in range(self.train_steps_per_t):
				samples = self.rb.get_random_batch()

				samples['tds'] = np.random.randint(0,self.max_td,size = self.rb._batch_size)

				losses = self._train(samples,[self.Q_Loss,self.R_Loss,self.F_Loss])
		else:
			losses = False
		return losses

	def add_sample(self,action,current_obs,next_obs,reward,done):
		self.rb.add_sample(action,current_obs,next_obs,reward,done)


			 