import numpy as np
import CGmaps


from variables import OptVar 

import maps

import matrix_aux_functions as mf

class Constraint:
	""" Each constraint is an expression of the form
	A*a + B*b + C*c +... + H 
	where (A,B,C) are maps, (a,b,c) are variables, and H is a constant (problem data)
	Primal constraints are equality constraits (Aa+Bb==0), dual constraints are inequalities (>=0).
	"""
	
	primal_constraints = []
	dual_constraints = []
	
	def __init__(self, label,
					sign_list,
					map_list,
					adj_flag_list,
					var_list,
					primal_or_dual,
					conjugateVar = None, 
					add_to_constr_list = True,
					const = None):
		
		assert len(map_list) == len(var_list), f'Supplied Maps and variables do not match in length in constraint {label}'
		assert len(sign_list) == len(var_list), f'Supplied sign_list and map_list do not match in length in constraint {label}'
		self.label = label # constraint name
		self.signs = sign_list
		self.maps = map_list
		self.adj_flags = adj_flag_list
		self.var_list = var_list
		self.conjugateVar = conjugateVar
 		
		if not const is None:
			self.const = const
		else:
			self.const = np.zeros(conjugateVar.matdim**2)
				
		# primal or dual constraint
		assert primal_or_dual in ('primal', 'dual'), \
			f"!!!!!!!!!!!!!!! Constraint {self.name} was not defined \n" + \
			'!!!!!!!!!!!!!!! Specify if constraint is *primal* or *dual*'
		
		self.primal_or_dual = primal_or_dual
		
		# to which list to add
		if primal_or_dual == 'primal': 
			constr_list = Constraint.primal_constraints
		else:
			constr_list = Constraint.dual_constraints
		
		if add_to_constr_list:
			constr_list.append(self)
		

	# @profile
	def __call__(self, v_in, v_out):
		'''
		Each constraint is an expression of the form \sum_i M_i(v_i) for some maps M_i and variables v_i.
		To each constraint ther is an associated conjugate variable. 
		When a (primal or dual) constraint is called, the expression \sum_i M_i(v_i) is added to the the conjugate var
		I.E. CONSTRAINTS ALWAYS ACT IN PLACE  as +=
		'''
		if maps.Maps.verbose:
			print(f"calling constraint {self.label}")
		
		if not self.conjugateVar.added_to_var_list:
			print('~~~~~~~~~~ !!!!!!!!!!!!!!!!! \n',\
			f'the variable conjugate to constraint {self.label} ({self.conjugateVar.name})', \
			'was not added to the list of variables and therefore does not have a buffer assigned' )
			return 1
		
		 
		if maps.Maps.verbose:
			print(f"conj var: {self.conjugateVar.name}")
		
		for s,M,f,var in zip(self.signs, self.maps, self.adj_flags, self.var_list):
			if maps.Maps.verbose:
				print(f"s = {s}")
				print(f"M = {M.name}")
				print(f"adj = {f}")
				print(f"var = {var}")
				print(f"out var slice = {v_out[ self.conjugateVar.slice ].shape}")
				 
			
			 
			# maps act in place as += 
			M.__call__( var, v_in, sign = s, adj_flag = f,  out = v_out[ self.conjugateVar.slice ]) 
			
			
			
		return 0
	
	@classmethod
	def print_constr_list(cls):
		for i, constr_list in enumerate((cls.primal_constraints, cls.dual_constraints)):
			print('='*80)
			print('~'*25 + ' {} CONSTRAINTS: '.format( ('PRIMAL', 'DUAL')[i]).center(20) + '~'*25 )
			print(f'Total of {len(constr_list)} constraints')
			print('='*80)
			print('name'.ljust(12) +' : ' + 'conj var'.ljust(12) + ' : ' + 'constr type'.ljust(12) + ' : expression') 
			print('-'*80)
			
			for c in constr_list:
				cls.print_constraint(c)
		
	
	@classmethod
	def print_constraint(cls, c):
			"""
			print the expression for the constraint
			"""
			print(f"{c.label:12} : {c.conjugateVar.name:12} : ", end=' ')
			
			for (sign, CGmap, adj_flag, optVar) in zip(c.signs, c.maps, c.adj_flags, c.var_list):
				if adj_flag:
					print(f"{cls.signString(sign)} {CGmap.adj_name}({optVar.name}) ", end = '')
				else:
					print(f"{cls.signString(sign)} {CGmap.name}({optVar.name}) ", end = '')
			
			print('\n')


	@staticmethod
	def signString(x): 
		s = f"{x:+}"
		if abs(x) == 1:
			s = s[0]
		return s
