import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import os
import copy
import json
from matplotlib import pyplot as plt
import datetime
import shutil

from agents import load_agent, load_expm_params
from dqn_agent import DQN_agent
from runner import Runner
import argparse

base_log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/logs/reinforcement_learning')
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser("Reinforcement experiment")
	parser.add_argument("expm_name", help="The name of the experiment to run", type=str)
	parser.add_argument("algorithm", help="The name of the algorithm to run", type=str)
	args = parser.parse_args()

	run_expm(args.algorithm,args.expm_name)

def setup_tf():
	tf.reset_default_graph() # THIS IS NECESSARY BEFORE MAKING NEW SESSION TO STOP IT ERRORING!!

	if not(tf.get_default_session() is None):
		tf.get_default_session().close()

	try:
		sess
	except:
		pass
	else:
		# sess.close()
		# del sess
		return sess
	sess = tf.InteractiveSession()

	return sess
				
def setup_logging(algorithm,expm_name,params,group=None):
	folder_name = '_'.join([algorithm,expm_name,datetime.datetime.now().strftime("%Y%m%d_%H%M%S")])
	log_dir = os.path.join(base_log_dir,folder_name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	params['log_dir'] = log_dir
	params['folder_name'] = folder_name
	
	# Save params to log dir
	with open(os.path.join(log_dir,'params.json'), 'w') as outfile:
		json.dump(params, outfile)

	return params

def clear_logs():
	base_log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/logs/reinforcement_learning')
	if os.path.exists(base_log_dir):
		shutil.rmtree(base_log_dir)

def setup_expm(algorithm,expm_name,params = None,restore_model_path=None,episode_finished_callback = None,persistent_to_print = [], use_AWS =False):

	sess = setup_tf()

	if params is None:
		params = load_expm_params(algorithm,expm_name)
	else:
		params = copy.deepcopy(params)

	params = setup_logging(algorithm,expm_name,params)
	
	params['use_AWS'] = use_AWS

	env = gym.make(params['env_name'])
	n_inputs = env.observation_space.shape[0]
	n_outputs = env.action_space.n
	
	agent = load_agent(algorithm,n_inputs,n_outputs,**copy.deepcopy(params))

	runner = Runner(env,agent,episode_finished_callback=lambda x : episode_finished_callback(x,persistent_to_print),**params)

	if not(restore_model_path is None):
		tf.reset_default_graph() 
		ckpt = tf.train.get_checkpoint_state(restore_model_path)
		if ckpt and ckpt.model_checkpoint_path:
			runner.saver.restore(sess, ckpt.model_checkpoint_path)
			# imported_meta = tf.train.import_meta_graph(restore_model) 
		print("Model restored.")

	return runner

def episode_finished_callback(runner):
	if runner.episodes % 10 == 0:
		print('Average reward at episode %d : %d' % (runner.episodes,np.mean(runner.episode_rewards[-100:])))
		print('Avg sample time is %f (ms)' % (1000*runner.avg_sample_time))
		print('Avg train time is %f (ms)' % ((runner.global_trains/runner.global_t)*(1000*runner.avg_train_time)))
						
		plt.figure(figsize=[4,3])
		plt.plot(runner.episode_rewards)
		plt.xlabel('Episode')
		plt.ylabel('Rewards')
		plt.show()
		plt.close()
		
def run_expm(algorithm,expm_name,params = None, runner=None, episode_finished_callback = None,persistent_to_print=[],use_AWS =False,restore_model_path=None):
	if runner is None:
		runner = setup_expm(algorithm,expm_name,params = None,episode_finished_callback = episode_finished_callback,persistent_to_print=persistent_to_print,use_AWS =use_AWS,restore_model_path=restore_model_path)

	runner.run()
	if not(runner.log_dir is None):
		print('Finished experiment, logged in: %s' % runner.log_dir)

	return runner

def grid_search(algorithm,expm_name,grid_params,params=None,runs_per_point = 3):
	
	if params is None:
		params = load_expm_params(algorithm,expm_name)
	else:
		params = copy.deepcopy(params)

	to_test = grid_params.items()
	to_test_keys = [val[0] for val in to_test]
	test_pts = np.meshgrid(*[val[1] for val in to_test])
	test_pts = np.reshape(np.transpose(test_pts),[-1,len(test_pts)])

	max_episodes = params['runner_params']['max_episodes']
	rewards = np.zeros((len(test_pts),runs_per_point,max_episodes))

	for i,tp in enumerate(test_pts):
		test_params = copy.deepcopy(params)
		
		for key,val in zip(to_test_keys,tp):
			test_params['agent_params'][key]= val
			print('Testing %s: %.2e' % (key,val))
		for j in range(runs_per_point):
			runner = run_expm(algorithm,expm_name, params = test_params)
			rewards[i,j] = runner.episode_rewards

	title = ', '.join(to_test_keys)
	labels = [', '.join(["%.2e" % val for val in pt]) for pt in test_pts]

	x = range(np.shape(rewards)[-1])
	ys = np.mean(rewards,axis=1)
	errors = np.std(rewards,axis=1)

	for y,error,lr in zip(ys,errors,labels):
		plt.plot(x, y, '-',label=lr)
		plt.fill_between(x, y-error, y+error,alpha=0.2)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.legend()
	plt.show()
	plt.close()

	return rewards
	
def sweep_test_cases(algorithm,expm_name,test_case_params,params = None,runs_per_point = 3,episode_finished_callback=episode_finished_callback, use_AWS=False):
	
	if params is None:
		params = load_expm_params(algorithm,expm_name)
	else:
		params = copy.deepcopy(params)

	folder_name = '_'.join(['sweep_test_cases',datetime.datetime.now().strftime("%Y%m%d_%H%M%S")])
	log_dir = os.path.join(base_log_dir,folder_name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	with open(os.path.join(log_dir,'test_case_params.json'), 'w') as outfile:
		json.dump(test_case_params, outfile)

	max_episodes = params['runner_params']['max_episodes']
	rewards = np.zeros((len(test_case_params),runs_per_point,max_episodes))
	
			
	for j in range(runs_per_point):
			
		for i,(test_name,test) in enumerate(iter(test_case_params.items())):
			test_params = copy.deepcopy(params)
			for cat, items in iter(test.items()):
				if cat == 'network_spec':
					test_params[cat] = items
				else:
					for item, values in iter(items.items()):
						test_params[cat][item] = values

				if not((j == 0) and (i == 0)):
					persistent_to_print = [('Testing %s, run %d' % (test_name,j)),make_sweep_fig(test_case_params.keys(),rewards[:,:(j+1),:])]
				else:
					persistent_to_print = [('Testing %s, run %d' % (test_name,j))]
				
				
				runner = run_expm(algorithm,expm_name, params = test_params,episode_finished_callback= episode_finished_callback, persistent_to_print=persistent_to_print,use_AWS=use_AWS)
				rewards[i,j] = runner.episode_rewards

				rewards.tofile(os.path.join(log_dir,'rewards.txt'), sep="\t", format="%s")

			

	if use_AWS:
		uf.aws_save_to_bucket(self.log_dir,self.params['folder_name'])

	display(make_sweep_fig(test_case_params.keys(),rewards))

	return rewards

def make_sweep_fig(labels,rewards):
	x = range(np.shape(rewards)[-1])
	ys = np.mean(rewards,axis=1)
	errors = np.std(rewards,axis=1)

	fig = plt.figure()
	for y,error,lr in zip(ys,errors,labels):
		plt.plot(x, y, '-',label=lr)
		plt.fill_between(x, y-error, y+error,alpha=0.2)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.legend()

	return fig
	


	