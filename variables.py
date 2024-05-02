
class OptVar:
	all_vars = {'primal': dict(), 'dual': dict()}
	
	def __init__(self, name, primal_or_dual, dims = ()):
		self.name = name
		self.primal_or_dual = primal_or_dual
		self.dims = dims
		try:
			OptVar.all_vars[primal_or_dual].update({name : self})
		except KeyError:
			print(f"!!!!!!!!!!!!!!! Variable {self.name} was not defined")
			print( '!!!!!!!!!!!!!!! Specify if varialbe is primal or dual')
			raise
		else:
			print(f"Variable {self.name} was added to the list of {primal_or_dual} variables")
			
		
	def print_var_list(self):
		for p_or_d in ('primal', 'dual'): 
			print(f'{p_or_d} vars:', end= '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
			for var_name, var in OptVar.all_vars[p_or_d].items():
				print(f"{var_name:20} with dims {var.dims}")
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

