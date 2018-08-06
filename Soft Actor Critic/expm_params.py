expm_params = {

	
	'CartPole-v1' : 
	{'base' : dict(
		env_name = 'CartPole-v1',
		epoch_length = 400,
		max_epochs = 5000,
		max_episode_length = -1,
		online_training = False,
		grad_steps_per_t = 1,
		),
	 'replay_buffer' : dict(
		batch_size = 32,
		max_buffer_size = 2000,
		min_pool_size = 32,
	 ),
	 'algorithm' : dict(
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
	 'nnet' : dict(
		n_hidden = 24,
		n_layers = 2
	 )
	},
	

		 'LunarLander-v2' : 
	{'base' : dict(
		env_name = 'LunarLander-v2',
		epoch_length = 1000,
		max_epochs = 100,
		online_training = False,
		grad_steps_per_t = 1,
		),
	 'replay_buffer' : dict(
		batch_size = 256,
		max_buffer_size = 1e6,
		min_pool_size = 1000,
	 ),
	 'algorithm' : dict(
		lr = 1e-5,
		tau = 0.01
	 ),
	 'policy' : dict(
		reward_scale = 1,
		epsilon_start = 1.0,
		epsilon_end = 0.1,
		epsilon_decay = 20000,
		scheme = 'Epsilon',
		),
	 'nnet' : dict(
		n_hidden = 20,
		n_layers = 2
	 )
	}
	
	
}