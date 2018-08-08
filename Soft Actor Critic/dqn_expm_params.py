expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.001,
		discount = 0.99,
		dueling = True,
        huber_loss = True,
        clip_gradients =  3.0,
        train_steps_per_t = 1
	 ),
	 'policy' : dict(
		epsilon_start = 1.0,
		epsilon_end = 0.01,
		epsilon_decay = 0.999,
		action_choice = 'Epsilon',
	 ),
	 'replay_buffer' : dict(
		batch_size = 32,
		max_size = 2000,
		min_pool_size = 100,
	 ),
	 'runner' : dict(
	 	max_episodes = 200,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=32)]
	},

	'CartPole_Soft' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		tau = 0.01,
		discount = 0.99,
		dueling = False,
        huber_loss = False,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        soft_learning = True,
        reward_scale = 1.0
	 ),
	 'policy' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer' : dict(
		batch_size = 32,
		max_size = 2000,
		min_pool_size = 100,
	 ),
	 'runner' : dict(
	 	max_episodes = 400,
		max_episode_length = 300,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=32)]
	},
	'LunarLander_Soft' : 
	{
	 'env_name' : 'LunarLander-v2',
	 'agent' : dict(
		lr = 1e-3,
		tau = 0.01,
		discount = 0.99,
		dueling = False,
        huber_loss = False,
        clip_gradients =  3.0,
        train_steps_per_t = 1,
        soft_learning = True,
        reward_scale = 1.0
	 ),
	 'policy' : dict(
		action_choice = 'Boltzmann',
	 ),
	 'replay_buffer' : dict(
		batch_size = 32,
		max_size = 2000,
		min_pool_size = 100,
	 ),
	 'runner' : dict(
	 	max_episodes = 200,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=32)]
	}
	}
