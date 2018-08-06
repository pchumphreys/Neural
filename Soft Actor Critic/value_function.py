import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from mlp import MLP

class Qnet(MLP):
	# Make a simple q network
	def __init__(self,action_size,obs,n_hidden,n_layers):
		# Ok so Q function takes s,a, gives Q   
		self.obs = obs
		# Super is used to call the init method of the parent class
		super(Qnet,self).__init__('qNet',self.obs,action_size,n_hidden,n_layers)
	  
class conv_Qnet():
	def __init__(self,action_size,obs,n_hidden):
		self._name = 'qNet'
		self.output_size = action_size
		self.n_hidden = n_hidden

		self.make_network(obs)

	def make_network(self,inputs = False,reuse = tf.AUTO_REUSE):
		# This function just makes a simple fully connected network. It is structured in a little bit of a silly way. The idea is that this lets one reuse the network weights elsewhere with different inputs. Currently not actually using this functionality 
		if inputs is False :
			inputs = self.inputs
			
		with tf.variable_scope(self._name,reuse = reuse):
			if not(isinstance(inputs,tf.Tensor)):  # Can chuck in more than one input. This just concatenates them
				inputs = tf.concat(inputs,axis=1)

		#The network recieves a frame from the game, flattened into an array.
		#It then resizes it and processes it through four convolutional layers.
		self.imageIn = tf.reshape(self.input,shape=[-1,84,84,3])
		self.conv1 = slim.conv2d( \
			inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
		self.conv2 = slim.conv2d( \
			inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
		self.conv3 = slim.conv2d( \
			inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
		self.conv4 = slim.conv2d( \
			inputs=self.conv3,num_outputs=hidden_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
		
		#We take the output from the final convolutional layer and split it into separate advantage and value streams.
		self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
		self.streamA = slim.flatten(self.streamAC)
		self.streamV = slim.flatten(self.streamVC)
		xavier_init = tf.contrib.layers.xavier_initializer()
		self.AW = tf.Variable(xavier_init([self.n_hidden//2,self.output_size]))
		self.VW = tf.Variable(xavier_init([self.n_hidden//2,1]))
		self.Advantage = tf.matmul(self.streamA,self.AW)
		self.Value = tf.matmul(self.streamV,self.VW)
		
		#Then combine them together to get our final Q-values.
		self.output = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
	 