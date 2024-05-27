import numpy as np
from matrix_aux_functions import dim_symm_matrix, dim_AntiSymm_matrix

complex_or_real = {True:'complex', False:'real'}


class OptVar:
	
	primal_vars = []
	dual_vars = []
	
	
	def __init__(self, name, primal_or_dual, dims, complex_var = True, add_to_var_list = True):
		self.name = name
		self.primal_or_dual = primal_or_dual
		self.dims = dims
		self.complex = complex_var
		
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
			
			if not var_list:
				last_index = -1
			else:
				if var_list[-1].complex:
					last_index = var_list[-1].indices_imag[-1] 
				else:
					last_index = var_list[-1].indices_real[-1] 
			
			if np.prod(dims)==1:
				self.indices_real = [last_index +1, last_index +1]
				if self.complex:
					print(f'!!!!!!!!!!!!!! variable {self.name} was set as complex but has dim=1 -->> set to be real')
					self.complex = False
				
			else:
				 # inds for real part
				self.indices_real = [last_index +1, last_index + dim_symm_matrix(np.prod(dims)) ]
				 # inds for imaginary part
				if self.complex:
					last_index = self.indices_real[-1]
					self.indices_imag = [last_index +1, last_index + dim_AntiSymm_matrix(np.prod(dims))]
				else:
					self.indices_imag = None
					
					
			
			# add variable to list
			var_list.append(self)			
			
			print(f"{complex_or_real[complex_var]} variable {self.name} was added to the list of {primal_or_dual} variables")
			
	
	# def get_inds(self):
		# '''
		# retriev indices for real and imaginary part
		# '''
		
		# if np.prod(self.dims) == 1:
			# real_inds = [in_var.indices[0] , in_var.indices[0]+1]
		# else:
			# real_inds = [in_var.indices[0], in_var.indices[1]]
		
		# if self.complex:
			# imag_inds = [in_var.indices[2]:in_var.indices[3]]
		# else:
			# imag_inds = None
		# return (real_inds, imag_inds)
	
				
	def print_var_list(self):
		print('='*80)
		print('~'*30 + ' VARIABLES: '.center(20) + '~'*30 )
		print('='*80)
		
		for i, var_list in enumerate((OptVar.primal_vars, OptVar.dual_vars)):
			print('~'*10 + " {} VARS:".format(('PRIMAL', 'DUAL')[i]))
			for var in var_list:
				print(f"{var.name:20} Inds real: {var.indices_real}", end=' ')
				if var.complex:
					print(f"{var.name:20} Inds_imag: {var.indices_imag}", end=' ')
								
				print(f": dims: {var.dims}: {complex_or_real[var.complex]}")
	

