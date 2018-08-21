expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.001,
		discount = 0.99,
		double = True,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        soft_learning = False,
        reward_scale = 2.0,
        multi_step = 1,
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
		lr = 1e-3,
		tau = 0.01,
		discount = 0.99,
		double = False,
        huber_loss = True,
        clip_gradients =  5.0,
        train_steps_per_t = 1,
        soft_learning = True,
        reward_scale = 1.0,
        multi_step = False,
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 64,
		max_size = 100000,
		min_pool_size = 1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 200,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=20)]
	},


	'Acrobot' : 
	{
	 'env_name' : 'Acrobot-v1-v2',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.01,
		discount = 0.99,
		double = False,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        soft_learning = True,
        reward_scale = 1.0
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 64,
		max_size = 100000,
		min_pool_size = 1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 3000,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=20)]
	},
	
	'Breakout' : 
	{
	 'env_name' : 'Breakout-ram-v0',
	 'agent_params' : dict(
		lr = 1e-4,
		tau = 0.0001,
		discount = 0.99,
		double = False,
        soft_learning = True,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        reward_scale = 1.0
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 64,
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

