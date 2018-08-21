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

            self.policy_outputs = tf.nn.softmax(self.outputs,axis=1) # Automatically sum to one.
            self.log_policy_outputs = tf.log(self.policy_outputs)
            self.optimal_policy = tf.argmax(self.policy_outputs,axis=1)      
            self.draw_policy_from_boltz_probs = tf.squeeze(tf.multinomial(self.outputs,1))


    def get_actions(self,obs,optimal_action = False):
        
        if optimal_action:
            return self.optimal_policy.eval(feed_dict = {self.inputs : [obs]})[0]

        else:
            if self.action_choice == 'Boltzmann':

                action = self.draw_policy_from_boltz_probs.eval(feed_dict = {self.inputs : [obs]})
                if action == self.action_size:
                    print('Numerical instability!')
                    print(self.outputs.eval(feed_dict = {self.inputs : [obs]}))
                    action -= 1
                return action
                    
            elif self.action_choice == 'Epsilon':

                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand() <= self.e:
                    action = np.random.randint(0,self.action_size)
                else:
                    action = self.optimal_policy.eval(feed_dict = {self.inputs : [obs]})

                if self.e > self.e_end:
                    self.e *= self.e_decay
                
                return action


class Policy_Discrete_for_Qnet(Base_Policy_Discrete):
    def __init__(self,net,**params):
        self.pnet = net
        super(Policy_Discrete_for_Qnet,self).__init__(self.pnet.inputs,self.pnet.outputs,self.pnet.output_size,**params)

class Policy_Discrete(Base_Policy_Discrete,MLP):
    def __init__(self,obs,action_size,layer_spec,scope='Policy',**params):

        MLP.__init__(self,scope,obs,action_size,layer_spec)
        Base_Policy_Discrete.__init__(self,obs,self.outputs,action_size,**params)
      
