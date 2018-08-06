import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Policy_Discrete():
    # Make a policy.
    # For now assume that discrete action space, such that tractable - obviously this slightly defeats the point of SAC implementation since all about how hard to compute the partition function
    # P function takes s, needs to be able to give actions.

    def __init__(self,Qnet,scheme = 'Bltz',reward_scale = 1.0,epsilon_start = 1,epsilon_end=0.1,epsilon_decay=0.99):

        self.Qnet = Qnet
        self.action_size = Qnet.output_size
        self.reward_scale = reward_scale
        self.scheme = scheme
        self.e = epsilon_start
        self.e_end = epsilon_end
        self.e_decay = epsilon_decay
        
        self.discrete = True
        self._name = 'Policy'
        self.make_policy_outputs(reuse=False)
        
    def make_policy_outputs(self, reuse = tf.AUTO_REUSE):
       
        with tf.variable_scope(self._name,reuse = reuse):
            if self.scheme == 'Bltz':
                self.policy_output = tf.nn.softmax(reward_scale*self.Qnet.output,axis=1) # Automatically sum to one.
                self.log_policy_output = tf.log(self.policy_output)
                self.action = tf.multinomial(self.log_policy_output, num_samples=1)[0] # Will generate an action
            elif self.scheme == 'Epsilon':
                self.action = tf.argmax(self.Qnet.output)
                
            
    def get_action(self,obs):
        
        if self.scheme == 'Bltz':
                a = self.action.eval(feed_dict = {self.Qnet.obs : [obs]})[0]
                
        elif self.scheme == 'Epsilon':
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand() <= self.e:
                a = np.random.randint(0,self.action_size)
            else:
                a = self.action.eval(feed_dict = {self.Qnet.obs : [obs]})[0]
            
            if self.e > self.e_end:
                self.e *= self.e_decay
            
        return a