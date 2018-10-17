expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		discount = 0.99,
        clip_gradients =  3.0,
        t_steps_per_train = 50,
        entropy_weight = 0.01,
        clip_param = 0.2
	 ),
	 'runner_params' : dict(
	 	max_episodes = 500,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=32)]
	},
	
	'LunarLander' : 
	{
	 'env_name' : 'LunarLander-v2',
	 'agent_params' : dict(
		lr = 1e-3,
		discount = 0.99,
        clip_gradients =  3.0,
        t_steps_per_train = 50,
        entropy_weight = 0.01,
        clip_param = 0.2
	 ),
	 'runner_params' : dict(
	 	max_episodes = 3000,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=64)]
	}
	}
