import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from mlp import MLP

class Base_Policy_Discrete():
    # Base policy function for discrete action space
    # P function takes s, needs to be able to give actions.

    def __init__(self,inputs,outputs,action_size,**params):

        self.inputs = inputs
        self.outputs = outputs
        self.action_size = action_size

        self.action_choice = params.pop('action_choice','Epsilon')
        self.e = params.pop('Epsilon_start',1.0)
        self.e_end = params.pop('Epsilon_end',0.01)
        self.e_decay = params.pop('Epsilon_decay',0.999)
        
        self.discrete = True

        self.make_policy_outputs(reuse=False)
        
    def make_policy_outputs(self, reuse = tf.AUTO_REUSE):

        if self.action_choice == 'Boltzmann':
            self.policy_outputs = tf.nn.softmax(self.outputs,axis=1) # Automatically sum to one.
            self.log_policy_outputs = tf.log(self.policy_outputs)
            self.action = tf.multinomial(self.log_policy_outputs, num_samples=1)[0] # Will generate an action
        elif self.action_choice == 'Epsilon':
            self.action = tf.argmax(self.outputs)
                
            
    def get_action(self,obs):
        
        if self.action_choice == 'Boltzmann':
                action = self.action.eval(feed_dict = {self.inputs : [obs]})[0]
                
        elif self.action_choice == 'Epsilon':
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand() <= self.e:
                action = np.random.randint(0,self.action_size)
            else:
                action = self.action.eval(feed_dict = {self.inputs : [obs]})[0]

            if self.e > self.e_end:
                self.e *= self.e_decay
            
        return action

class Policy_Discrete_for_Qnet(Base_Policy_Discrete):
    def __init__(self,net,**params):
        self.pnet = net
        super(Policy_Discrete_for_Qnet,self).__init__(self.pnet.inputs,self.pnet.outputs,self.pnet.output_size,**params)

class Policy_Discrete(Base_Policy_Discrete,MLP):
    def __init__(self,obs,action_size,layer_spec,**params):

        MLP.__init__(self,'Policy',obs,action_size,layer_spec)
        Base_Policy_Discrete.__init__(self,obs,self.outputs,action_size,**params)
      
