expm_params = {
	'CartPole' : 
	{
	 'env_name' : 'CartPole-v1',
	 'agent_params' : dict(
		lr = 1e-3,
		discount = 0.99,
        clip_gradients =  3.0,
        t_steps_per_train = 30,
        entropy_weight = 0.01
	 ),
	 'runner_params' : dict(
	 	max_episodes = 400,
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
        clip_gradients =  0.5,
        t_steps_per_train = 100,
        entropy_weight = 0.01,
        vf_weight = 1.0
	 ),
	 'runner_params' : dict(
	 	max_episodes = 500,
		max_episode_length = -1,
	  ),
	 'network_spec' : 2*[dict(type = 'dense',size=64)]
	}
	}
