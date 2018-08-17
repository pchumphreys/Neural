import copy

def load_expm_params(algorithm,expm_name):
	if algorithm == 'A2C':
		from a2c_expm_params import expm_params
		return _load_params(algorithm,expm_name,expm_params)
	elif algorithm == 'PPO':
		from ppo_expm_params import expm_params
		return _load_params(algorithm,expm_name,expm_params)
	elif algorithm == 'DQN':
		from dqn_expm_params import expm_params
		return _load_params(algorithm,expm_name,expm_params)
	elif algorithm == 'TDM':
		from tdm_expm_params import expm_params
		return _load_params(algorithm,expm_name,expm_params)
	else:
		raise NotImplementedError(algorithm)

def _load_params(algorithm,expm_name,expm_params):
	if expm_name in expm_params:
		return copy.deepcopy(expm_params[expm_name])
	else: 
		raise NotImplementedError(algorithm,expm_name)

def load_agent(algorithm,*args,**kargs):
	if algorithm == 'A2C':
		from a2c_agent import A2C_agent
		return A2C_agent(*args,**kargs)
	elif algorithm == 'PPO':
		from ppo_agent import PPO_agent
		return PPO_agent(*args,**kargs)
	elif algorithm == 'DQN':
		from dqn_agent import DQN_agent
		return DQN_agent(*args,**kargs)
	elif algorithm == 'TDM':
		from tdm_agent import TDM_agent
		return TDM_agent(*args,**kargs)

	else:
		raise NotImplementedError(algorithm)
