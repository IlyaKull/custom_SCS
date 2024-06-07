import numpy as np
from matrix_aux_functions import dim_symm_matrix, dim_AntiSymm_matrix

class OptVar:
	
	primal_vars = []
	dual_vars = []
	
	
	def __init__(self, name, primal_or_dual, dims, cone = 'Rn', dtype = complex, add_to_var_list = True):
		self.name = name
		self.primal_or_dual = primal_or_dual
		self.dims = dims
		self.dtype = dtype
		self.added_to_var_list = add_to_var_list
		
		assert cone in {'Rn','PSD'}, f"Specify cone to which variable {self.name} belongs: 'Rn' or 'PSD'"
		self.cone = cone
		
		if add_to_var_list:
			# primal or dual variable
			assert primal_or_dual in ('primal', 'dual'), \
				f"!!!!!!!!!!!!!!! Variable {self.name} was not defined \n" + \
				'!!!!!!!!!!!!!!! Specify if varialbe is primal or dual'
			
			# to which list to add
			if primal_or_dual == 'primal': 
				var_list = OptVar.primal_vars
			else:
				var_list = OptVar.dual_vars
			
						
			self.matrix = np.zeros((np.prod(self.dims) ,np.prod(self.dims) ) , dtype = self.dtype)
					
			
			# add variable to list
			var_list.append(self)			
			
			print(f"{dtype.__name__} variable {self.name} was added to the list of {primal_or_dual} variables")
			
	
				
	def print_var_list(self):
		print('='*80)
		print('~'*30 + ' VARIABLES: '.center(20) + '~'*30 )
		print('='*80)
		
		for i, var_list in enumerate((OptVar.primal_vars, OptVar.dual_vars)):
			print('~'*10 + " {} VARS:".format(('PRIMAL', 'DUAL')[i]))
			for var in var_list:
				print(f"{var.name:20} : matrix base: {var.matrix.base}", end=' ')
				print(f": dims: {var.dims}: {var.dtype} : cone = {var.cone}")
	

