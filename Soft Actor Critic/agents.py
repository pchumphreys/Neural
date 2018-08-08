def load_expm_params(algorithm,expm_name):
	if algorithm == 'A2C':
		from a2c_agent_params import expm_params
		_load_params(expm_params,algorithm,expm_name)
	elif algorithm == 'DQN':
		from dqn_agent_params import expm_params
		_load_params(expm_params,algorithm,expm_name)
	else:
		raise NotImplementedError(algorithm)

def _load_params(expm_params,algorithm,expm_name):
	if expm_name in expm_params:
			return expm_params[expm_name]
		else: 
			raise NotImplementedError(algorithm,expm_name)

def load_agent(algorithm,*args,**kargs):
	if algorithm == 'A2C':
		from a2c_agent import A2C_Agent
		return A2C_Agent(*args,**kargs)
	elif algorithm == 'DQN':
		from dqn_agent import DQN_Agent
		return DQN_Agent(*args,**kargs)
	else:
		raise NotImplementedError(algorithm)
