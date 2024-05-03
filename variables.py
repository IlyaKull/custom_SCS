import numpy as np

class OptVar:
	
	primal_vars = []
	dual_vars = []
	
	
	def __init__(self, name, primal_or_dual, dims):
		self.name = name
		self.primal_or_dual = primal_or_dual
		self.dims = dims
		
		# primal or dual variable
		assert primal_or_dual in ('primal', 'dual'), \
			f"!!!!!!!!!!!!!!! Variable {self.name} was not defined \n" + \
			'!!!!!!!!!!!!!!! Specify if varialbe is primal or dual'
		
		# to which list to add
		if primal_or_dual == 'primal': 
			var_list = OptVar.primal_vars
		else:
			var_list = OptVar.dual_vars
		
		if not var_list:
			last_index = -1
		else:
			last_index = var_list[-1].indices[1] 
		
		self.indices = (last_index +1, last_index + dim_symm_matrix(np.prod(dims)) ) 
		
		# add variable to list
		var_list.append(self)			
		print(f"Variable {self.name} was added to the list of {primal_or_dual} variables")
			
				
	def print_var_list(self):
		print('='*80)
		print('~'*30 + ' VARIABLES: '.center(20) + '~'*30 )
		print('='*80)
		
		for i, var_list in enumerate((OptVar.primal_vars, OptVar.dual_vars)):
			print('~'*10 + " {} VARS:".format(('PRIMAL', 'DUAL')[i]))
			for var in var_list:
				print(f"{var.name:20} Indices: {var.indices[0] :<8d} -- {var.indices[1] : <16d} : dims: {var.dims} ")
		
		

def dim_symm_matrix(d):
	return int((d**2 + d)/2)
	
