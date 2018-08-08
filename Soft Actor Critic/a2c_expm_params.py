expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent' : dict(
		lr = 1e-3,
		discount = 0.95,
        clip_gradients =  3.0,
        t_steps_per_train = 30,
        entropy_weight = 0.01
	 ),
	 'policy' : dict(
		reward_scale = 1.0,
	 ),
	 'runner' : dict(
	 	max_episodes = 400,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=32)]
	}
	}
