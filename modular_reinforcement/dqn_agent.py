import tensorflow as tf
import numpy as np

import util_functions as uf
from base_agent import Base_Agent
from value_functions import Qnet
from policies import Policy_Discrete_for_Qnet
from memories import Replay_Buffer

class DQN_agent(Base_Agent):
	# This class handles the training of the networks
	def __init__(self,n_inputs,n_outputs,**params):

		super(DQN_agent,self).__init__(**params)
		
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs

		self.lr = params['agent_params'].pop('lr',1e-3)
		self.discount = params['agent_params'].pop('discount',1e-3)
		self.tau = params['agent_params'].pop('tau',1e-3)

		self.soft_learning = params['agent_params'].pop('soft_learning',False)
		self.reward_scale = params['agent_params'].pop('reward_scale',1.0)
		self.double =  params['agent_params'].pop('double', False)
		self.huber_loss =  params['agent_params'].pop('huber_loss', True)
		self.clip_gradients =  params['agent_params'].pop('clip_gradients', False)
		self.train_steps_per_t = params['agent_params'].pop('train_steps_per_t',1)
		self.multi_step = params['agent_params'].pop('multi_step',False)
		if self.multi_step:
			self.discount = self.discount**self.multi_step

		assert not(self.soft_learning and self.double)
		
		self.image_obs = params['agent_params'].pop('image_obs',False)
		self.image_buffer_frames  = params['agent_params'].pop('image_buffer_frames',1)
		if self.image_obs:
			params['replay_buffer_params']['image_obs'] = True
			params['replay_buffer_params']['image_buffer_frames'] = self.image_buffer_frames
			self.n_inputs = [80,80,self.image_buffer_frames]
			self.frames_since_done = 0
			self.current_obs_frame_cat = np.zeros(self.n_inputs)

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

		self.rb = Replay_Buffer(self.n_inputs,self.n_outputs,discrete_action=True,multi_step = self.multi_step ,**params['replay_buffer_params'])
	
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
		
		self.train_ops = []
		self._init_qnet_training_op()
		
		self.target_Q_update = uf.update_target_network(self.model_Q_params,self.target_Q_params,tau=self.tau,update_op_control_dependencies=self.train_ops)
		self.train_ops.append(self.target_Q_update)


		self._finish_agent_setup()



	def _init_placeholders(self):

		self.actions = tf.placeholder(tf.float32,shape = [None,self.n_outputs],name = 'actions')
		self.obs = tf.placeholder(tf.float32,shape = [None]+self.n_inputs,name = 'observations')
		self.next_obs = tf.placeholder(tf.float32,shape = [None]+ self.n_inputs,name = 'next_observations')
		self.rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
		self.dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')

		self.scaled_rewards = self.reward_scale * self.rewards
		
	def _construct_feed_dict(self,samples):
		return {self.actions : samples['actions'],
				self.obs : samples['obs'],
				self.next_obs : samples['next_obs'],
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
	
					

	def train(self):
		if self.rb.batch_ready():
			for j in range(self.train_steps_per_t):
				samples = self.rb.get_random_batch()
				if self.multi_step:
					samples['rewards'] = uf.calc_discount(samples['rewards'],self.discount,axis=1)[:,0]
				losses = self._train(samples,[self.Q_Loss])
		else:
			losses = False
		return losses

	def add_sample(self,action,current_obs,next_obs,reward,done):
		if self.image_obs:
			self.add_image_sample(action,current_obs,next_obs,reward,done)
		else:
			self.rb.add_sample(action,self.pre_process_obs(current_obs),self.pre_process_obs(next_obs),reward,done)


	def add_image_sample(self,action,current_obs,next_obs,reward,done):
		
		self.frames_since_done += 1
		if self.frames_since_done == 1:
			self.current_obs_frame_cat[:,:,-1] = np.squeeze(self.current_obs)

		next_obs_frame_cat = np.concatenate((self.current_obs_frame_cat[:,:,1:],uf.preprocess_image_obs(next_obs)),axis=2)
		if self.frames_since_done >= self.image_buffer_frames:
			self.rb.add_sample(action,self.current_obs_frame_cat,next_obs_frame_cat,reward,done)
		self.current_obs_frame_cat = next_obs_frame_cat
		if done:
			self.frames_since_done = 0

	def pre_process_obs_for_action(self,obs):
		if self.image_obs:
			if self.frames_since_done >= self.image_buffer_frames:
				return self.current_obs_frame_cat # already has obs from last step
			elif self.frames_since_done == 0:
				self.current_obs = uf.preprocess_image_obs(obs) # process here, save for add image sample
				return np.tile(self.current_obs,[1,1,self.image_buffer_frames])
			else: 
				
				return np.concatenate((np.tile(np.expand_dims(self.current_obs_frame_cat[:,:,-(self.frames_since_done+1)],axis=2),[1,1,(self.image_buffer_frames - self.frames_since_done)])
					,self.current_obs_frame_cat[:,:,-(self.frames_since_done):]),axis=2)
		else:
			return obs




			 