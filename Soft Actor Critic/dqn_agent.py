import tensorflow as tf
import numpy as np

from agent import Agent
from value_functions import Qnet
from policies import Policy_Discrete
from replay_buffer import Replay_Buffer

class DQN_agent(Agent):
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


        self.qnet = Qnet(n_outputs,self.obs,params['network_spec'])
        self.model_Q_params = self.qnet.get_params_internal()
        self.model_Q_outputs = self.qnet.outputs

        # Duplicate the Qnet with different variables for the target network
        with tf.variable_scope('qNet_T'):
            self.target_Q_outputs = self.qnet.make_network(inputs = self.next_obs,reuse=False) 
            self.target_Q_params = self.qnet.get_params_internal()

        self.model_Q_predict_from_next_obs = tf.stop_gradient(tf.one_hot(tf.argmax(self.qnet.make_network(inputs = self.next_obs),axis=1),self.qnet.output_size))
        
        
        self.policy = Policy_Discrete(self.qnet,**params['policy'])

        self.rb = Replay_Buffer(n_inputs,n_outputs,discrete_action=True,**params['replay_buffer'])
    
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        
        self.train_ops = []
        self.init_Q_net_training()
        self.init_target_Q_update()

        self._finish_agent_setup()
        

    def init_Q_net_training(self):
        

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
            gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
        self.train_Q = self.optimizer.apply_gradients(zip(gradients, variables))

        self.train_ops.append(self.train_Q)
        
    def init_target_Q_update(self):
        
        with tf.variable_scope('Target_Q_update'):
            self.target_Q_update = []
            for tQ_p in self.target_Q_params:
                # Match each target net param with equiv from vnet
                Q_p = [v for v in self.model_Q_params if tQ_p.name[(tQ_p.name.index('/')+1):] in v.name]
                assert(len(Q_p) == 1) # Check that only found one variable
                Q_p = Q_p[0]
                with tf.control_dependencies([self.train_Q]):
                    self.target_Q_update.append(tQ_p.assign(self.tau * Q_p + (1-self.tau)*tQ_p))
            self.target_Q_update = tf.group(self.target_Q_update)
            
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

             