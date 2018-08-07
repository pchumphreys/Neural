import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import os

from expm_params import expm_params
from algorithms import Deep_Q_Learning
from runner import Runner
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser("deep_q_expm")
	parser.add_argument("expm_name", help="The name of the experiment to run", type=str)
	args = parser.parse_args()

	run_expm(args.expm_name)

def setup_tf():
	tf.reset_default_graph() # THIS IS NECESSARY BEFORE MAKING NEW SESSION TO STOP IT ERRORING!!
	try:
		sess
	except:
		pass
	else:
		sess.close()
		del sess
	sess = tf.InteractiveSession()

	return sess
                

def run_expm(expm_name, params = None):

	sess = setup_tf()

	if params is None:
		params = expm_params[expm_name]

	log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
								   'tensorflow/logs/deep_q_learning',expm_name)
	params['log_dir'] = log_dir
	
	env = gym.make(params['env_name'])
	n_inputs = env.observation_space.shape[0]
	n_outputs = env.action_space.n
	
	agent = Deep_Q_Learning(n_inputs,n_outputs,params)

	runner = Runner(env,agent,params['runner'])

	runner.run()
 



	