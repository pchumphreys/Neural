import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import os
import copy
import pickle

from agents import load_agent, load_expm_params
from dqn_agent import DQN_agent
from runner import Runner
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser("Reinforcement experiment")
	parser.add_argument("expm_name", help="The name of the experiment to run", type=str)
	parser.add_argument("algorithm", help="The name of the algorithm to run", type=str)
	args = parser.parse_args()

	run_expm(args.expm_name,args.algorithm)

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
                
def episode_finished_callback(runner):
	if runner.episodes % 100 == 0:
		print('Average reward at episode %d : %d' % (runner.episodes,np.mean(runner.episode_rewards[-100:])))

def run_expm(expm_name,algorithm,params = None):

	sess = setup_tf()

	if params is None:
		params = load_expm_params(algorithm,expm_name)
	else:
		params = copy.deepcopy(params)

	log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
								   'tensorflow/logs/reinforcement_learning',algorithm,expm_name)
	params['log_dir'] = log_dir
	pickle.dump(params, open(os.path.join(log_dir,'params.pickle'),'wb'))

	print('logging in %s:' % log_dir)

	env = gym.make(params['env_name'])
	n_inputs = env.observation_space.shape[0]
	n_outputs = env.action_space.n
	
	agent = load_agent(algorithm,n_inputs,n_outputs,**params)

	saver = tf.train.Saver()

	runner = Runner(env,agent,episode_finished_callback=episode_finished_callback,saver=saver,**params)

	runner.run()

	sess.close()

	return runner
 



	