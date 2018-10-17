expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 3e-4,
		tau = 0.001, # IMPORTANT TO NOT INCREASE TOO FAR, 0.001 is stable
		discount = 0.99,
		max_td = 1,
		double = False,
		soft_learning = True,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        q_train_steps_per_t = 4,
        reward_scale = 2.0
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
	 'network_spec' : 2*[dict(type = 'dense',size=32,reg_weight=0.001)]
	},

	'LunarLander' : 
	{
	 'env_name' : 'LunarLander-v2',
	 'agent_params' : dict(
		lr = 3e-4,
		tau = 0.001,
		discount = 0.99,
		max_td = 3,
		double = False,
        soft_learning = True,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        q_train_steps_per_t = 1,
        reward_scale = 5.0
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 128,
		max_size = 200000,
		min_pool_size =1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 500,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=64,reg_weight=0.001)]
	},
	'Breakout' : 
	{
	 'env_name' : 'Breakout-ram-v0',
	 'agent_params' : dict(
		lr = 1e-4,
		tau = 0.001,
		discount = 0.99,
		max_td = 1,
		double = False,
        soft_learning = True,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        q_train_steps_per_t = 1,
        reward_scale = 1.0
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 128,
		max_size = 1000000,
		min_pool_size =1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 10000,
		max_episode_length = -1,
	  ),
	 'network_spec' : 4*[dict(type = 'dense',size=128,reg_weight=0.001)]
	},

	}

