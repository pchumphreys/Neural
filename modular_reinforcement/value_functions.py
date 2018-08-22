import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_hub as hub
import numpy as np

from mlp import MLP

class Qnet(MLP):
	# Make a simple q network
	def __init__(self,obs,action_size,layer_spec,scope='qNet'):
		# Ok so Q function takes s, gives Q(a|s) 
		# Super is used to call the init method of the parent class
		super(Qnet,self).__init__(scope,obs,action_size,layer_spec)
	  
class Qnet_advantage(MLP):
	def __init__(self,obs,action_size,layer_spec,scope='qNet'):

		super(Qnet_advantage,self).__init__(scope,obs,action_size,layer_spec,layer_callbacks = {'adv_value' : adv_value_layer},final_linear_layer=False)
	  
	def adv_value_layer(self,inputs,layer_number,**layer_spec):
		scope = layer_spec.pop('scope','fc_' + str(layer_number))
		with tf.variable_scope(scope):
			streamA,streamV = tf.split(inputs,2,0)
			adv = slim.fully_connected(streamA,self.output_size,activation_fn=None)
			val = slim.fully_connected(streamV,1,activation_fn=None)
			return val + adv - tf.reduce_mean(adv,axis=1,keep_dims=True)


class Qnet(MLP):
	# Make a simple q network
	def __init__(self,obs,action_size,layer_spec,scope='qNet'):
		# Ok so Q function takes s, gives Q(a|s) 
		# Super is used to call the init method of the parent class
		super(Qnet,self).__init__(scope,obs,action_size,layer_spec)

