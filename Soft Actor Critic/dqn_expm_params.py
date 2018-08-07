expm_params = {
	'CartPole' : 
	dict(
		env_name ='CartPole-v1',
		epoch_length = 400,
		max_epochs = 5000,
		max_episode_length = -1,
		online_training = False,
		grad_steps_per_t = 1,
		),
	 'agent' : dict(
		lr = 1e-3,
		tau = 0.001,
		discount = 0.95,
	 ),
	 'policy' : dict(
		reward_scale = 1,
		epsilon_start = 1.0,
		epsilon_end = 0.01,
		epsilon_decay = 0.99,
		scheme = 'Epsilon',
	 ),
	 'replay_buffer' : dict(
		batch_size = 32,
		max_buffer_size = 2000,
		min_pool_size = 32,
	 ),
	 'network_spec' : 2*[dict(type = 'dense',size=24)]
	},

}