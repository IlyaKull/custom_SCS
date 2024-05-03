
class OptVar:
	all_vars = {'primal': dict(), 'dual': dict()}
	
	def __init__(self, name, primal_or_dual, dims):
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
		print('~'*30 + ' VARIABLES: '.center(20) + '~'*30 )
		print('='*80)
		for p_or_d in ('primal', 'dual'): 
			print('~'*20 + f' {p_or_d} vars:')
			print('-'*80)
			for var_name, var in OptVar.all_vars[p_or_d].items():
				print(f"{var_name:20} dims {var.dims}")
			print('-'*80)
		print('='*80)
