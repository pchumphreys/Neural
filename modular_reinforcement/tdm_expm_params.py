expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.005,
		discount = 0.99,
		max_td = 1,
		double = False,
		soft_learning = False,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        q_train_steps_per_t = 1,
        reward_scale = 3.0
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 64,
		max_size = 10000,
		min_pool_size = 1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 400,
		max_episode_length = 300,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=16,reg_weight=None)]
	},

	'LunarLander' : 
	{
	 'env_name' : 'LunarLander-v2',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.005,
		discount = 0.99,
		max_td = 3,
		double = False,
        soft_learning = False,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        q_train_steps_per_t = 1,
        reward_scale = 2.0
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 64,
		max_size = 200000,
		min_pool_size =1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 400,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=32)]
	},

	}

