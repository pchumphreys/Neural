import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Policy_Discrete():
    # Make a policy.
    # For now assume that discrete action space, such that tractable - obviously this slightly defeats the point of SAC implementation since all about how hard to compute the partition function
    # P function takes s, needs to be able to give actions.

    def __init__(self,qnet,**params):

        self.qnet = qnet
        self.action_size = self.qnet.output_size
 
        self.scheme = params.pop('scheme','Epsilon')
        self.e = params.pop('Epsilon_start',1.0)
        self.e_end = params.pop('Epsilon_end',0.01)
        self.e_decay = params.pop('Epsilon_decay',0.999)
        self.reward_scale = params.pop('reward_scale',1.0)
        
        self.discrete = True
        self._name = 'Policy'
        self.make_policy_outputs(reuse=False)
        
    def make_policy_outputs(self, reuse = tf.AUTO_REUSE):
       
        with tf.variable_scope(self._name,reuse = reuse):
            if self.scheme == 'Boltzmann':
                self.policy_output = tf.nn.softmax(reward_scale*self.qnet.outputs,axis=1) # Automatically sum to one.
                self.log_policy_output = tf.log(self.policy_output)
                self.action = tf.multinomial(self.log_policy_output, num_samples=1)[0] # Will generate an action
            elif self.scheme == 'Epsilon':
                self.action = tf.argmax(self.qnet.outputs)
                
            
    def get_action(self,obs):
        
        if self.scheme == 'Boltzmann':
                action = self.action.eval(feed_dict = {self.qnet.inputs : [obs]})[0]
                
        elif self.scheme == 'Epsilon':
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand() <= self.e:
                action = np.random.randint(0,self.action_size)
            else:
                action = self.action.eval(feed_dict = {self.qnet.inputs : [obs]})[0]
            
            if self.e > self.e_end:
                self.e *= self.e_decay
            
        return action