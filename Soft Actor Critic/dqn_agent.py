import tensorflow as tf
import numpy as np

from base_agent import Base_Agent
from value_functions import Qnet
from policies import Policy_Discrete_for_Qnet
from replay_buffer import Replay_Buffer

class DQN_agent(Base_Agent):
    # This class handles the training of the networks
    def __init__(self,n_inputs,n_outputs,**params):

        super(DQN_agent,self).__init__(n_inputs,n_outputs,**params)
      
        self.lr = params['agent'].pop('lr',1e-3)
        self.discount = params['agent'].pop('discount',1e-3)
        self.tau = params['agent'].pop('tau',1e-3)
        self.dueling =  params['agent'].pop('dueling', True)
        self.huber_loss =  params['agent'].pop('huber_loss', True)
        self.clip_gradients =  params['agent'].pop('clip_gradients', False)
        self.train_steps_per_t = params['agent'].pop('train_steps_per_t',1)

        self.init_placeholders()

        self.qnet = Qnet(self.obs,n_outputs,params['network_spec'],scope='qnet')
        self.model_Q_params = self.qnet.get_params_internal()
        self.model_Q_outputs = self.qnet.outputs
        self.model_Q_predict_from_next_obs = tf.stop_gradient(tf.one_hot(tf.argmax(self.qnet.make_network(inputs = self.next_obs),axis=1),self.qnet.output_size))
        
        # Duplicate the Qnet with different variables for the target network
        self.tqnet = Qnet(self.next_obs,n_outputs,params['network_spec'],scope='tqnet')
        self.target_Q_outputs = self.tqnet.outputs 
        self.target_Q_params = self.tqnet.get_params_internal()
        
        self.policy = Policy_Discrete_for_Qnet(self.qnet,**params['policy'])

        self.rb = Replay_Buffer(n_inputs,n_outputs,discrete_action=True,**params['replay_buffer'])
    
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        
        self.train_ops = []
        self.init_qnet_training_op()
        self.init_target_qnet_update()

        self._finish_agent_setup()
    
    def init_placeholders(self):

        self.actions = tf.placeholder(tf.float32,shape = [None,n_outputs],name = 'actions')
        self.obs = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'observations')
        self.next_obs = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'next_observations')
        self.rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
        self.dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')

     def _construct_feed_dict(self,samples):
        return {self.actions : samples['actions'],
                self.obs : samples['observations'],
                self.next_obs : samples['next_observations'],
                self.dones : samples['dones'],
                self.rewards : samples['rewards']}
   

    def init_qnet_training_op(self):
        

        with tf.variable_scope('Q_loss'):

            if not(self.dueling):
                self.Q_t = tf.stop_gradient(self.rewards +  self.discount * (1-self.dones) * tf.reduce_max(self.target_Q_outputs,axis=1))
            else:
                self.Q_t = tf.stop_gradient(self.rewards +  self.discount * (1-self.dones) * tf.reduce_sum(self.target_Q_outputs*self.model_Q_predict_from_next_obs,axis=1))
        
            if self.huber_loss:
                self.Q_Loss = tf.reduce_mean(tf.sqrt(1+tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - self.Q_t))-1)
            else:
                self.Q_Loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.model_Q_outputs*self.actions,axis=1) - self.Q_t))

            tf.summary.scalar('Q_loss', self.Q_Loss)
        

        Qnet_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self.qnet._name)
        Qnet_regularization_loss = tf.reduce_sum(
            Qnet_regularization_losses)
    
        gradients, variables = zip(*self.optimizer.compute_gradients(self.Q_Loss + Qnet_regularization_loss,var_list = self.model_Q_params))
        if self.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_gradients)
        self.train_Q = self.optimizer.apply_gradients(zip(gradients, variables))

        self.train_ops.append(self.train_Q)
        
    def init_target_qnet_update(self):
                 
        self.target_Q_update = util_functions.update_target_network(self.model_Q_params,self.target_Q_params,tau=self.tau,update_op_control_dependencies=[self.train_Q])
        self.train_ops.append(self.target_Q_update)

                    

    def train(self):
        if self.rb.batch_ready():
            for j in range(self.train_steps_per_t):
                samples = self.rb.get_samples()
                losses = self._train(samples,self.Q_Loss)
        else:
            losses = False
        return losses

    def add_sample(self,action,current_obs,next_obs,reward,done):
        self.rb.add_sample(action,current_obs,next_obs,reward,done)

    def get_action(self,obs):
        return self.policy.get_action(obs)



             