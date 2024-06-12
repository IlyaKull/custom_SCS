import CGmaps
from variables import OptVar 
import numpy as np
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
					var_list,
					primal_or_dual,
					constr_type,
					constant = None,
					conjugateVar = None, 
					add_to_constr_list = True):
		
		assert len(map_list) == len(var_list), f'Supplied Maps and variables do not match in length in constraint {label}'
		assert len(sign_list) == len(var_list), f'Supplied sign_list and map_list do not match in length in constraint {label}'
		self.label = label # constraint name
		self.signs = sign_list
		self.maps = map_list
		self.var_list = var_list
		self.constr_type = constr_type
						
		try:
			assert all( m.dims['out'] == self.maps[0].dims['out'] for m in self.maps ), \
				f'Output dimension mismatch in constraint "{label}"'
		except AssertionError as xxx:
			print('!!!!!!!!!!!!! output dims:')
			for m in self.maps:
				print(m.dims['out'])
			raise xxx

		if conjugateVar: 
			self.conjugateVar = conjugateVar
			assert self.conjugateVar.matdim == self.maps[0].dims['out'] , f'Dimension mismatch between constraint {label} and dual variable {conjugateVar.name}'
		else:
			self.conjugateVar = OptVar(f'MISSING VAR:{label}:','dual', dims =  self.maps[0].dims['out'], add_to_var_list = False)
	
		self.constant = constant
		
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
		
		assert constr_type in ('EQ', 'PSD','OBJ'), \
			f"!!!!!!!!!!!!!!! Constraint {self.label} was not defined \n" + \
			'!!!!!!!!!!!!!!! Specify: *EQ* for equality constraint, *PSD* for positivity, or *OBJ* for objective function'
		
		self.constr_type = constr_type
		
		if add_to_constr_list:
			constr_list.append(self)
		
	
	
	def __call__(self, v_in, v_out, add_to_out = False):
		'''
		Each constraint is an expression of the form \sum_i M_i(v_i) for some maps M_i and variables v_i.
		To each constraint ther is an associated conjugate variable. 
		When a (primal or dual) constraint is called, the expression \sum_i M_i(v_i) is written to the the conjugate var
		add_to_out specifies whether out = expr or out = out + expr
		'''
		# print(f"v_in shape = {v_in.shape}")
		# print(f"v_out shape = {v_out.shape}")
		if not self.conjugateVar.added_to_var_list:
			print('~~~~~~~~~~ !!!!!!!!!!!!!!!!! \n',\
			f'the variable conjugate to constraint {self.label} ({self.conjugateVar.name})', \
			'was not added to the list of variables and therefore does not have a buffer assigned' )
			return 1
		
		if not add_to_out:
			v_out[ self.conjugateVar.slice ] = np.zeros((self.conjugateVar.matdim,)*2, dtype = self.conjugateVar.dtype ).ravel()
		
			if maps.Maps.verbose:
				print(f"conj var: {self.conjugateVar.name}")
		
		for s,M,var in zip(self.signs, self.maps, self.var_list):
			if maps.Maps.verbose:
				print(f"s = {s}")
				print(f"M = {M.name}")
				print(f"var = {var}")
				
				print(f"out var slice = {v_out[ self.conjugateVar.slice ].shape}")
				print(f"result size = {(	s * M.__call__( var, v_in ) ).ravel().shape}")
			
			v_out[ self.conjugateVar.slice ] += (	s * M.__call__( var, v_in ) ).ravel()
		
		if self.constant:
			v_out[ self.conjugateVar.slice ] += \
				(\
				self.constant * np.identity( self.conjugateVar.matdim)\
				).ravel()
		
		return 0
	
		
	def print_constr_list(self):
		for i, constr_list in enumerate((Constraint.primal_constraints, Constraint.dual_constraints)):
			print('='*80)
			print('~'*25 + ' {} CONSTRAINTS: '.format( ('PRIMAL', 'DUAL')[i]).center(20) + '~'*25 )
			print(f'Total of {len(constr_list)} constraints')
			print('='*80)
			print('name'.ljust(12) +' : ' + 'conj var'.ljust(12) + ' : ' + 'constr type'.ljust(12) + ' : expression') 
			print('-'*80)
			for c in constr_list:
				print_constraint(c)
		
	

def print_constraint(c):
		"""
		print the expression for the constraint
		"""
		print(f"{c.label:12} : {c.conjugateVar.name:12} : {c.constr_type:12} : ", end=' ')
		
		for (sign, CGmap, optVar) in zip(c.signs, c.maps, c.var_list):
			if optVar is None:
				print(f"{signString(sign)} {CGmap.name}*1 ", end = '')
			else:
				print(f"{signString(sign)} {CGmap.name}({optVar.name}) ", end = '')
		
		if c.constant:
			print(c.constant, end = ' ')
		
		print('\n')

def signString(x): 
	s = f"{x:+}"
	if abs(x) == 1:
		s = s[0]
	return s
