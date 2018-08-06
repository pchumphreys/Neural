class logger():
	def __init__(self):
		self.reset()
		
	def reset(self):
		self.vars = {}
		
	def record(self,var_name,value):
		if var_name in self.vars:
			self.vars[var_name].append(value)
		else:
			self.vars[var_name] = [value]
	
	def keys():
		return self.vars.keys()

	def get(self,var_name):
		if var_name in self.vars:
			return self.vars[var_name]
		else:
			return False