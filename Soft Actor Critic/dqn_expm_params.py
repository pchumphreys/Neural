expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent' : dict(
		lr = 1e-3,
		tau = 0.001,
		discount = 0.95,
		dueling = False,
        huber_loss = True,
        clip_gradients =  False,
        train_steps_per_t = 1
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
	 'runner' : dict(
	 	max_episodes = 400,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=24)]
	}
	}
