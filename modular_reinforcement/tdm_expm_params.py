expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.01,
		max_td = 3,
		discount = 0.99,
        huber_loss = False,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        reward_scale = 1.0
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 32,
		max_size = 2000,
		min_pool_size = 100,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 400,
		max_episode_length = 300,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=16)]
	},


	}

