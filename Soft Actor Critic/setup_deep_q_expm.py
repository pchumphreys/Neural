import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import os


from logger import logger
from value_function import Qnet
from expm_params import expm_params
from sampler import Sampler
from policy import Policy_Discrete
from algorithms import Deep_Q_Learning
from replay_buffer import Replay_Buffer
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser("deep_q_expm")
	parser.add_argument("expm_name", help="The name of the experiment to run", type=str)
	args = parser.parse_args()

	writer,merged,sampler,algo,qnet,policy,run_params = setup_expm(args.expm_name)

	run_expm(writer,merged,sampler,algo,qnet,policy,run_params)

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


def setup_expm(expm_name, params = expm_params):

	sess = setup_tf()
	log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
								   'tensorflow/logs/soft_actor_critic')

	base_params = params[expm_name]['base']
	if base_params['env_name'] == 'GridWorld':
		from gridworld import gameEnv
		env = gameEnv(partial=False,size=5)
		n_inputs = 21168
		n_outputs = env.actions
	else:
		env = gym.make(base_params['env_name'])
		n_inputs = env.observation_space.shape[0]
		n_outputs = env.action_space.n
	
	epoch_length = base_params['epoch_length']
	max_epochs = base_params['max_epochs']
	max_episode_length = base_params['max_episode_length']
	online_training = base_params['online_training']
	grad_steps_per_t = base_params['grad_steps_per_t']
	run_params = {'epoch_length' : epoch_length, 'max_epochs' : max_epochs, 'online_training' : online_training, 'grad_steps_per_t' : grad_steps_per_t}
	
	lr = params[expm_name]['algorithm']['lr']
	tau = params[expm_name]['algorithm']['tau']
	discount = params[expm_name]['algorithm']['discount']

	reward_scale = params[expm_name]['policy']['reward_scale']
	epsilon_start = params[expm_name]['policy']['epsilon_start']
	epsilon_end = params[expm_name]['policy']['epsilon_end']
	epsilon_decay = params[expm_name]['policy']['epsilon_decay']
	policy_scheme = params[expm_name]['policy']['scheme']

	n_hidden = params[expm_name]['nnet']['n_hidden']
	n_layers = params[expm_name]['nnet']['n_layers']

	batch_size = params[expm_name]['replay_buffer']['batch_size']
	max_buffer_size = params[expm_name]['replay_buffer']['max_buffer_size']
	min_pool_size = params[expm_name]['replay_buffer']['min_pool_size']
	# Todo make these into lists so that can define each layer separately


	rewards = tf.placeholder(tf.float32,shape = [None],name = 'rewards')
	actions = tf.placeholder(tf.float32,shape = [None,n_outputs],name = 'actions')
	observations = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'observations')
	next_observations = tf.placeholder(tf.float32,shape = [None,n_inputs],name = 'next_observations')
	dones = tf.placeholder(tf.float32,shape = [None],name = 'dones')

	if base_params['env_name'] == 'GridWorld':
		qnet = conv_Qnet(n_outputs,observations,n_hidden)
	else:
		qnet = Qnet(n_outputs,observations,n_hidden,n_layers)

	policy = Policy_Discrete(qnet,reward_scale=reward_scale,epsilon_start = epsilon_start,epsilon_end=epsilon_end,epsilon_decay=epsilon_decay,scheme=policy_scheme)

	algo = Deep_Q_Learning(qnet,actions,observations,next_observations,rewards,dones,lr=lr,tau=tau,discount=discount)

	log = logger()

	rb = Replay_Buffer(n_inputs,n_outputs,max_buffer_size,min_pool_size = min_pool_size,batch_size=batch_size)
	sampler = Sampler(policy,env,rb,log,max_episode_length=max_episode_length)
 
	merged = tf.summary.merge_all()

	writer = tf.summary.FileWriter(log_dir, sess.graph)

	tf.global_variables_initializer().run()

	return log,writer,merged,sampler,algo,qnet,policy,run_params

def run_expm(log,writer,merged,sampler,algo,qnet,policy,run_params):
	for i in range(run_params['max_epochs']):
		sampler.reset()
		epoch_avg_losses = 0
		
		for t in range(run_params['epoch_length']):
			sampler.sample()
			
			if sampler.batch_ready() or run_params['online_training']:
				if run_params['online_training']:
					samples = sampler.get_last_sample()
					summary,losses,qnet_o = algo.train(samples,merged,algo.Q_Loss,qnet.output) 
				else:
					for j in range(run_params['grad_steps_per_t']):
						samples = sampler.get_samples()
						summary,losses= algo.train(samples,merged,algo.Q_Loss) 
				epoch_avg_losses = (epoch_avg_losses*(t) + np.array(losses))/(t+1)
				   
		log.record('mean_episode_reward',sampler.mean_episode_reward)
		writer.add_summary(summary, i)
		print(epoch_avg_losses)
		
		writer.flush()

		print('Epoch %i, mean_reward %d' % (i, sampler.mean_episode_reward))
	  
