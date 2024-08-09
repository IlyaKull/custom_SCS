import numpy as np
import logging

logger = logging.getLogger(__name__)


class OptVar:
	
	primal_vars = []
	dual_vars = []
	
	len_primal_vec_y = 0
	len_dual_vec_x = 0
	lists_closed = False
	dtype = None
	
	x_slice = None
	y_slice = None
	tao_slice = None
 	
	def __init__(self, name, primal_or_dual, dims, cone = 'Rn', dtype = complex):
		
		assert not OptVar.lists_closed, '!!!!!!!!!!! cannot add variables because variable list is closed (OptVar.lists_closed = True)'
		
		self.name = name
		self.primal_or_dual = primal_or_dual
		self.dims = dims
		self.dtype = dtype
		# self.added_to_var_list = add_to_var_list
		self.matdim = np.prod(dims)
		
		assert cone in {'Rn','PSD'}, f"Specify cone to which variable {self.name} belongs: 'Rn' or 'PSD'"
		self.cone = cone
		
		# if add_to_var_list:
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
			last_s = 0
		else:
			last_s = var_list[-1].slice.stop
		
		
		# self.indices = [last_index +1, last_index + dim_symm_matrix(np.prod(dims)) ]
		self.slice = slice(last_s, last_s + self.matdim**2 )
								
		
		# add variable to list
		var_list.append(self)			
		
		logger.debug(f"{dtype.__name__} variable {self.name} was added to the list of {primal_or_dual} variables")
			
	
	@classmethod
	def _close_var_lists(cls):
		logger.debug('closing variable list (OptVar.lists_closed = True )')
		
		cls.len_dual_vec_x = cls.dual_vars[-1].slice.stop
		cls.len_primal_vec_y = cls.primal_vars[-1].slice.stop
		cls.lists_closed = True
		
		
		if logging.DEBUG >= logging.root.level:
			cls.print_var_list()
		
		# slice of all dual vars x in u=[x,y,tao]
		cls.x_slice = slice(cls.dual_vars[0].slice.start, cls.dual_vars[-1].slice.stop)
		cls.y_slice = slice(cls.x_slice.stop, cls.x_slice.stop + cls.primal_vars[-1].slice.stop) 
		cls.tao_slice = slice(cls.y_slice.stop, cls.y_slice.stop +1)
		 
		
		# determine dtype of x and y
		dtypes = [ v.dtype for v in cls.dual_vars + cls.primal_vars ]
		if complex in dtypes:
			cls.dtype = complex
		else:
			cls.dtype = float
		
		 
		
		logger.debug(f"length of primal vector (y): {cls.len_primal_vec_y}")
		logger.debug(f"length of dual vector (x): {cls.len_dual_vec_x}")
		logger.debug(f"dtype of x and y: {cls.dtype}")


	@classmethod
	def print_var_list(cls):
		print('='*100)
		print('~'*30 + ' VARIABLES: '.center(20) + '~'*30 )
		print('-'*100)
		
		for i, var_list in enumerate((cls.primal_vars, cls.dual_vars)):
			print('~'*10 + " {} VARS:".format(('PRIMAL', 'DUAL')[i]))
			for var in var_list:
				print(f"{var.name:20} : slice: {var.slice}", end=' ')
				print(f": dims: {var.dims}: {var.dtype} : cone = {var.cone}")

		print('='*100)







# define subclass of dict that allows only certain keys
# this is all just because I don't want to store k-body states omega_k in a_list[k]

import collections.abc

class DuplicateKeyError(KeyError):
	pass
	

class RestrictedKeysDict(collections.abc.MutableMapping, dict):

	def __init__(self, *args, allowed_keys,  **kwargs):
		self._dict = dict(*args, **kwargs)
		self.allowed_keys = allowed_keys
		
	def __getitem__(self, key):
		return self._dict[key]

	def __setitem__(self, key, value):
		if key not in self.allowed_keys:
			raise DuplicateKeyError(f"Key '{key}' is not a member of allowed keys: {self.allowed_keys}.")
		self._dict[key] = value

	def __delitem__(self, key):
		del self._dict[key]

	def __iter__(self):
		return iter(self._dict)

	def __len__(self):
		return len(self._dict)
	
	def __str__(self):
		return self._dict.__str__() + "_{dict with restricted keys: " + f"{self.allowed_keys}" + "}"
