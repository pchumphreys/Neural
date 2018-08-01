
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

#Model - this makes the network (ie the policy)

#we have the soft Q function, the parameterised value function V, and the parameterised tractable policy pi

# TODOs
# Make comments properly

# Make run time routine
# Whitening of inputs

# Need batching

# Need recall

# Train, test routines


# Here is a helper class to make a simple neural network. Importantly, it allows us to easily get the parameters, and hopefully to link the inputs to other variables
# The get_output functionality is borrowed from the SAC reference code.

class MLP():
    def init(name,inputs,size_out,n_hidden,n_layers):
        self._name = name
        self.output = tf.get_output(inputs,size_out,n_hidden,n_layers)
        
    def get_output(*inputs,size_out,n_hidden,n_layers,reuse = False):
        with tf.variable_scope(self._name,reuse = reuse):
            # To do understand weight initialization!   
            self.hidden = slim.stack(inputs, slim.fully_connected, [n_hidden]*n_layers, scope='fc',activation_fn=tf.nn.relu)
            outputs = slim.fully_connected(self.hidden,size_out)
        return outputs

    def get_params_internal(self):

        scope = tf.get_variable_scope().name
        scope += '/' + self._name + '/' if len(scope) else self._name + '/'

        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
        )

class Qnet():
    def init(self,s_size,a_size,n_hidden,n_layers):
        # Ok so Q function takes s,a, gives Q   
        self.in_a = tf.placeholder(tf.float32,shape = [None, size_a])
        self.in_s = tf.placeholder(tf.float32,shape = [None, size_s])
        super(Qnet,self).__init__('qNet',(self.in_a,self.in_s),1,n_hidden,n_layers)


class Vnet():
    def init(self,s_size,n_hidden,n_layers):
        # Ok so V function takes s, gives V
        self.in_s = tf.placeholder(tf.float32,shape = [None, size_s])
        super(Qnet,self).__init__('vNet',(self.in_a,self.in_s),1,n_hidden,n_layers)

class Policy_Discrete(MLP):
    # P function takes s, needs to be able to give actions.
    # For now assume that discrete action space, such that tractable - obviously this slightly defeats the point of SAC implementation since all about how hard to compute the partition function
    def init(self,s_size,a_size,n_hidden,n_layers):

        self.in_s = tf.placeholder(tf.float32,shape = [None, size_s])
        super(Policy_Discrete,self).__init__('policy',(self.in_s),a_size,n_hidden,n_layers)

    def get_output_for_obs(self,observations, ret_log_out = False, reuse = False):
        
        self.outputs = self.get_output(observations, reuse = reuse)
        
        with tf.variable_scope(self._name,reuse = reuse):
            self.softmax = tf.nn.softmax(self.outputs) # Automatically sum to one.
            self.log_outputs = tf.log(self.softmax)
            self.action = tf.multinomial(tf.log(self.softmax), num_samples=1)[0] # Will generate an action
        if ret_log_out:
            return self.outputs, self.log_outputs
        else:
            return self.outputs

    def get_actions(self):
        pass

#Ok, so pass the Pnet, Qnet, Vnet

class SAC():
    def init(self,Qnet,Vnet,Policy):
        self.rewards_placeholder = tf.placeholder(tf.float32,shape = [None])
        self.action_placeholder = tf.placeholder(tf.float32,shape = [None])
        self.obs_placeholder = tf.placeholder(tf.float32,shape = [None])
        self.next_obs_placeholder = tf.placeholder(tf.float32,shape = [None])

    def init_Q_net_training(self,lr):
        Qs = Qnet.get_outputs((self.action_placeholder,self.state_placeholder),reuse=True)

        with tf.variable_scope('target'):
            target_Vs = Vnet.get_outputs(self.state_placeholder,reuse=True)
            self.target_V_params = Vnet.get_params_internal

        self.Q_Loss = 0.5*tf.reduce_sum(tf.square(Qs - self.rewards_placeholder - self.discount * target_Vs))
        training_variables = Qnet.get_params_internal()

        self.Q_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.Q_optimizer.minimize(Q_Loss,var_list = training_variables)

    def init_V_net_training(self,lr):
        Vs = Qnet.get_outputs(self.state_placeholder,reuse=True)
        Qs = Qnet.get_outputs((self.action_placeholder,self.state_placeholder),reuse=True)

        V_Loss = 0.5*tf.reduce_sum(tf.square(Vs - Qs + Pnet.log_outputs))
        training_variables = Vnet.get_params_internal()

        self.V_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.V_optimizer.minimize(V_Loss,var_list = training_variables)


    def init_Policy_training(self,lr):
        policy_a, policy_log_a = Policy.get_output_for_obs(self.state_placeholder,ret_log_out = True, reuse=True)
        Vs = Qnet.get_outputs(self.state_placeholder,reuse=True)
        Qs = Qnet.get_outputs((self.action_placeholder,self.state_placeholder),reuse=True)

        P_Loss = 0.5*tf.reduce_sum(tf.square(policy_log_a - Qs + Vs))
        training_variables = Pnet.get_params_internal()
        
        self.P_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.P_optimizer.minimize(P_Loss,var_list = training_variables)



    def train(self):
        pass

# action = tf.clip_by_value(logits,tf.expand_dims(env.action_space.low,0),tf.expand_dims(env.action_space.high,0)) # Have to clip the action space. This might be a bad idea


#should make so that the pi can be easily changed

#Algorithm ie Soft Actor Critic - training etc makes the ops

#Env

#optimizer

class envMaker():
    def init(env_name):
        # Setup the gym environment!

    env_name = 'LunarLander-v2'
    env = gym.make(env_name)

    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.n