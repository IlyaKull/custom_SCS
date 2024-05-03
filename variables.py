
class OptVar:
	
	primal_vars = []
	dual_vars = []
	
	
	def __init__(self, name, primal_or_dual, dims):
		self.name = name
		self.primal_or_dual = primal_or_dual
		self.dims = dims
		
		
		# compute indices
		
		# Add variable to list of primal/dual variables
		assert primal_or_dual in ('primal', 'dual'), \
			f"!!!!!!!!!!!!!!! Variable {self.name} was not defined \n" + \
			'!!!!!!!!!!!!!!! Specify if varialbe is primal or dual'
				
		if primal_or_dual == 'primal': 
			OptVar.primal_vars.append(self)
		else:
			OptVar.dual_vars.append(self)
			
		print(f"Variable {self.name} was added to the list of {primal_or_dual} variables")
			
		
	def print_var_list(self):
		print('~'*30 + ' VARIABLES: '.center(20) + '~'*30 )
		print('='*80)
		
		for i, var_list in enumerate((OptVar.primal_vars, OptVar.dual_vars)):
			print('~'*10 + " {} VARS:".format(('PRIMAL', 'DUAL')[i]))
			for var in var_list:
				print(f"{var.name:20} dims {var.dims}")
		print('-'*80)
		print('='*80)
