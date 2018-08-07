import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from mlp import MLP

class Qnet(MLP):
	# Make a simple q network
	def __init__(self,action_size,obs,layer_spec):
		# Ok so Q function takes s, gives Q(a|s) 
		# Super is used to call the init method of the parent class
		super(Qnet,self).__init__('qNet',obs,action_size,layer_spec)
	  
class Qnet_advantage(MLP):
	def __init__(self,action_size,obs,layer_spec):

		super(Qnet,self).__init__('qNet',obs,action_size,layer_spec,layer_callbacks = ['adv_value' = adv_value_layer],final_linear_layer=False)
	  
	def adv_value_layer(self,inputs,**layer_spec):
		self.streamA,self.streamV = tf.split(inputs,2,0)
		self.adv = slim.fully_connected(self.hidden,self.output_size,activation_fn=None)
		self.val = slim.fully_connected(self.hidden,1,activation_fn=None)
		return self.val + self.adv - tf.reduce_mean(self.adv,axis=1,keep_dims=True)
