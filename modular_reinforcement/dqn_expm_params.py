expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.0001,
		discount = 0.99,
		double = False,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        soft_learning = True,
        reward_scale = 1.0,
        multi_step = 1,
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 64,
		max_size = 10000,
		min_pool_size = 100,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 400,
		max_episode_length = 300,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=8,reg_weight=0.0001)]
	},

	'LunarLander' : 
	{
	 'env_name' : 'LunarLander-v2',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.0001,
		discount = 0.99,
		double = False,
        huber_loss = True,
        clip_gradients =  2.0,
        train_steps_per_t = 1,
        soft_learning = True,
        reward_scale = 1.0,
        multi_step = False,
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 128,
		max_size = 100000,
		min_pool_size = 1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 200,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=20,reg_weight=0.01)]
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
	 'env_name' : 'Breakout-v4',
	 'atari_env' : True,
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.0001,
		discount = 0.99,
		double = False,
        soft_learning = True,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        reward_scale = 1.0,
        image_obs = True,
        image_buffer_frames = 2
	 ),
	 'policy_params' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer_params' : dict(
		batch_size = 128,
		max_size = 100000,
		min_pool_size =1000,
	 ),
	 'runner_params' : dict(
	 	max_episodes = 20000,
		max_episode_length = -1,
	  ),
	 'network_spec' : ([dict(type = 'conv2d', size = 32, kernel = [8,8],stride = 4),
	 				    dict(type = 'conv2d', size = 64, kernel = [4,4],stride = 2),
	 				    dict(type = 'conv2d', size = 64, kernel = [3,3],stride = 1),
	 				    dict(type = 'flatten')]
	 				  +[dict(type = 'dense',size=512,reg_weight=0.001)])
	},


	}

