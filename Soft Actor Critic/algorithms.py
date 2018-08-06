import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Deep_Q_Learning():
    # This class handles the training of the networks
    def __init__(self,Qnet,actions,obs,next_obs,rewards,dones,lr=3e-4,discount = 0.99, tau=0.005):
        self.lr = lr
        self.discount = discount
        self.tau = tau 
        
        # Maybe would be nicer to not pass these but define here, but this seems to be messy. Once check if works, could go back to defining here
        self.actions = actions
        self.obs = obs
        self.next_obs = next_obs
        self.rewards = rewards
        self.dones = dones
        
        self.Qnet = Qnet
        self.Q_outputs = Qnet.output
        
        # Duplicate the Qnet with different variables for the target network
        with tf.variable_scope('qNet_T'):
            self.target_Q_outputs = Qnet.make_network(inputs = self.next_obs,reuse=False) 
            self.target_Q_params = Qnet.get_params_internal()
        
        self.predict = tf.stop_gradient(tf.one_hot(tf.argmax(Qnet.make_network(inputs = self.next_obs),axis=1),self.Qnet.output_size))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        
        self.train_ops = []
        self.init_Q_net_training()
        self.init_target_Q_update()
        
        
    def init_Q_net_training(self):
        training_variables = self.Qnet.get_params_internal()
        with tf.variable_scope('Q_loss'):
            self.Q_t = tf.placeholder(tf.float32,[None], name = 'Loss')
            if True:
                self.in_Q_t = tf.stop_gradient(self.rewards +  self.discount * (1-self.dones) * tf.reduce_max(self.target_Q_outputs,axis=1))
            else:
                self.in_Q_t = tf.stop_gradient(self.rewards +  self.discount * (1-self.dones) * tf.reduce_sum(self.target_Q_outputs*self.predict,axis=1))
        
            if True:
                self.Q_Loss = tf.reduce_mean(tf.sqrt(1+tf.square(tf.reduce_sum(self.Q_outputs*self.actions,axis=1) - self.Q_t))-1)
            else:
                self.Q_Loss = tf.reduce_mean(tf.square(tf.reduce_sum(self.Q_outputs*self.actions,axis=1) - self.Q_t))

            tf.summary.scalar('Q_loss', self.Q_Loss)
        

#         Qnet_regularization_losses = tf.get_collection(
#             tf.GraphKeys.REGULARIZATION_LOSSES,
#             scope=self.Qnet._name)
#         Qnet_regularization_loss = tf.reduce_sum(
#             Qnet_regularization_losses)
    
#         gradients, variables = zip(*self.optimizer.compute_gradients(self.Q_Loss + Qnet_regularization_loss,var_list = training_variables))
#         gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#         self.train_Q = self.optimizer.apply_gradients(zip(gradients, variables))

        self.train_Q = self.optimizer.minimize(-self.Q_Loss,var_list = training_variables)

        self.train_ops.append(self.train_Q)
        
    def init_target_Q_update(self):
        # Pull the qnet params
        qnet_params = self.Qnet.get_params_internal()
        
        with tf.variable_scope('Target_Q_update'):
            self.tQnet_update = []
            for tQ_p in self.target_Q_params:
                # Match each target net param with equiv from vnet
                Q_p = [v for v in qnet_params if tQ_p.name[(tQ_p.name.index('/')+1):] in v.name]
                assert(len(Q_p) == 1) # Check that only found one variable
                Q_p = Q_p[0]
                with tf.control_dependencies([self.train_Q]):
                    self.tQnet_update.append(tQ_p.assign(self.tau * Q_p + (1-self.tau)*tQ_p))
            self.tQnet_update = tf.group(self.tQnet_update)
            
        self.train_ops.append(self.tQnet_update)
        
    def _construct_feed_dict(self,samples):  
        return {self.actions : samples['actions'],
                    self.obs : samples['observations'],
                    self.next_obs : samples['next_observations'],
                    self.dones : samples['dones'],
                    self.rewards : samples['rewards']}
                    
    def train(self, samples, *args):
        feed_dict = self._construct_feed_dict(samples)
        return tf.get_default_session().run([self.train_ops] + list(args), feed_dict = feed_dict)[1:]
