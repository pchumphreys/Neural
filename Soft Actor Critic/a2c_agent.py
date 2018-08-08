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

        super(A2C_agent,self).__init__(n_inputs,n_outputs,**params)
      
        self.lr = params['agent'].pop('lr',1e-3)
        self.discount = params['agent'].pop('discount',1e-3)
        self.entropy_weight = params['agent'].pop('entropy_weight',0.01)

        self.clip_gradients =  params['agent'].pop('clip_gradients', False)
        self.t_steps_per_train = params['agent'].pop('t_steps_per_train',1)
        
        self.init_placeholders()

        self.model_v = Qnet(self.obs,1,params['network_spec'])
        self.model_v_params = self.val_net.get_params_internal()
        self.model_v_outputs = self.val_net.outputs
        
        params['policy']['action_choice'] = params['policy'].get('action_choice',True)
        assert params['policy']['action_choice'] == 'Boltzmann' # Softmax on outputs
        
        self.policy = Policy_Discrete(self.obs,n_outputs,params['network_spec'],**params['policy'])
        self.policy_params = self.policy.get_params_internal()

        self.memory = Memory(n_inputs,n_outputs,discrete_action=True,max_size = self.t_steps_per_train)
    
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        
        self.train_ops = []
        self.init_val_training_op()
        self.init_policy_training_op()

        self._finish_agent_setup()
        
      def init_placeholders(self):

        self.actions = tf.placeholder(tf.float32,shape = [None,n_outputs],name = 'actions')
        self.obs = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'observations')
        self.val_ts = tf.placeholder(tf.float32,shape = [None], name = 'value_targets')
        self.advantages = tf.placeholder(tf.float32,shape = [None], name = 'advantages')
              
     def _construct_feed_dict(self,samples):
        return {self.actions : samples['actions'],
                self.obs : samples['observations'],
                self.val_ts : samples['value_targets'],
                self.advantages : samples['advantages']}
   

    def init_val_training_op(self):
        

        with tf.variable_scope('val_loss'):
            self.val_loss = 0.5*tf.reduce_sum(tf.square(self.model_v_outputs - self.val_t))
            tf.summary.scalar('val_loss', self.val_loss)

        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,scope=self.model_v._name)
        regularization_loss = tf.reduce_sum(
            regularization_losses)
    
        gradients, variables = zip(*self.optimizer.compute_gradients(self.val_loss + regularization_loss,var_list = self.model_val_params))
        if self.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_gradients)
        self.train_val = self.optimizer.apply_gradients(zip(gradients, variables))

        self.train_ops.append(self.train_val)
    
    def init_policy_training_op(self):

        with tf.variable_scope('policy_loss'):
            self.policy_output_for_action = tf.reduce_sum(self.policy.policy_outputs * self.actions,axis=1)
            self.raw_policy_loss = -tf.reduce_sum(tf.log(self.policy_output_for_action)*self.advantages)
            self.policy_entropy = -tf.reduce_sum(self.policy.policy_outputs * self.policy.log_policy_outputs)

            self.policy_loss = self.raw_policy_loss - self.entropy_weight * self.policy_entropy
            tf.summary.scalar('policy_loss', self.policy_loss)
        
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,scope=self.policy._name)
        regularization_loss = tf.reduce_sum(
            regularization_losses)
    
        gradients, variables = zip(*self.optimizer.compute_gradients(self.policy_loss + regularization_loss,var_list = self.policy_params))
        if self.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_gradients)
        self.train_policy = self.optimizer.apply_gradients(zip(gradients, variables))

       
        self.train_ops.append(self.train_policy)
        
    def train(self):
        if.memory.last_sample_done() or memory.is_full():
            samples = memory.get_all_samples()
            if memory.is_full():
                values = self.get_values(samples['obs'] + samples['next_obs'][-1])
                bootstrap = values[-1]
            else:
                values = self.get_values(samples['obs']) + [0]
                bootstrap = 0

            samples['value_targets']  = uf.calc_discount(samples['rewards'] + [bootstrap],self.discount)[:-1] #discounted_total_rewards
            # Generalised advantage estimation
            samples['advantages'] = uf.calc_discount(samples['rewards'] + self.discount * values[1:] - values[:-1])
            
            losses = self._train(samples,self.Q_Loss)
        else:
            losses = False
        return losses

    def get_values(self,obs):
        return self.model_v_outputs.eval(self.obs=obs)

    def add_sample(self,action,current_obs,next_obs,reward,done):
        self.memory.add_sample(action,current_obs,reward,done,next_obs)

    def get_action(self,obs):
        return self.policy.get_action(obs)

             