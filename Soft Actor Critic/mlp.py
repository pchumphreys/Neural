import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Here is a helper class to make a simple neural network. Importantly, it allows us to easily get the parameters, and hopefully to link the inputs to other variables
# The get_output functionality is borrowed from the SAC reference code.

class MLP():
	def __init__(self,name,inputs,output_size,n_hidden,n_layers):
		self._name = name
		self.inputs = inputs
		self.output_size = output_size
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		
		self.output = self.make_network(reuse = False)
		
	def make_network(self,inputs = False,reuse = tf.AUTO_REUSE):
		# This function just makes a simple fully connected network. It is structured in a little bit of a silly way. The idea is that this lets one reuse the network weights elsewhere with different inputs. Currently not actually using this functionality 
		if inputs is False :
			inputs = self.inputs
			
		with tf.variable_scope(self._name,reuse = reuse):
			if not(isinstance(inputs,tf.Tensor)):  # Can chuck in more than one input. This just concatenates them
				inputs = tf.concat(inputs,axis=1)

			# To do: understand weight initialization!   
			self.hidden = slim.stack(inputs, slim.fully_connected, [self.n_hidden]*self.n_layers, scope='fc',activation_fn=tf.nn.relu) #,weights_regularizer=slim.l2_regularizer(0.1)
			outputs = slim.fully_connected(self.hidden,self.output_size,activation_fn=None)
		return outputs

	def get_params_internal(self):
		# Useful function to get network weights
		
		scope = tf.get_variable_scope().name
		scope += '/' + self._name + '/' if len(scope) else self._name + '/'

		return tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
		)