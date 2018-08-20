import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Here is a helper class to make a simple neural network. Importantly, it allows us to easily get the parameters, and hopefully to link the inputs to other variables
# The get_output functionality is borrowed from the SAC reference code.

class MLP():
	def __init__(self,scope,inputs,output_size,layer_spec,final_linear_layer=True,layer_callbacks = []):
		self._name = scope
		self.inputs = inputs
		self.output_size = output_size
		self.layer_spec = layer_spec
		self.layer_callbacks = layer_callbacks
		self.final_linear_layer = final_linear_layer

		self.outputs = self.make_network(reuse = False)
		
	def make_network(self,inputs = False,reuse = tf.AUTO_REUSE):
		# This function just makes a simple fully connected network. It is structured in a little bit of a silly way. The idea is that this lets one reuse the network weights elsewhere with different inputs. Currently not actually using this functionality 
		if inputs is False :
			inputs = self.inputs
			
		with tf.variable_scope(self._name,reuse = reuse):
			if not(isinstance(inputs,tf.Tensor)):  # Can chuck in more than one input. This just concatenates them
				inputs = [inputt if tf.rank(inputt).eval() == 2 else tf.expand_dims(inputt,axis=1) for inputt in inputs]
				inputs = tf.concat(inputs,axis=1)

			outputs = inputs

			for i,layer in enumerate(self.layer_spec):
				if layer['type'] == 'dense':
					outputs = self.make_dense_layer(outputs,i,**layer)
				elif layer['type'] in self.layer_callbacks:
					outputs = self.layer_callbacks[layer['type']](outputs,i,**layer)
				else:
					raise NotImplementedError
			#,weights_regularizer=slim.l2_regularizer(0.1)
			
			if self.final_linear_layer:
				outputs = slim.fully_connected(outputs,self.output_size,activation_fn=None)# Final linear layer

		return outputs

	def get_params_internal(self):
		# Useful function to get network weights
		
		scope = tf.get_variable_scope().name
		scope += '/' + self._name + '/' if len(scope) else self._name + '/'

		return tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
		)

	def make_dense_layer(self,inputs,layer_number,**layer_spec):
		
		activation_fn = layer_spec.pop('activation_fn',tf.nn.relu)
		scope = layer_spec.pop('scope','fc_' + str(layer_number))
		size = layer_spec.pop('size')
		reg_weight = layer_spec.pop('reg_weight',None)
		regularizer = None if (reg_weight is None) else slim.l2_regularizer(reg_weight)
		return slim.fully_connected(inputs,size,scope=scope,activation_fn=activation_fn,weights_regularizer=regularizer)
		