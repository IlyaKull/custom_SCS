import CGmaps
from variables import OptVar 
import numpy as np
import maps

class Constraint:
	""" Each constraint is an expression of the form
	A*a + B*b + C*c +... + H 
	where (A,B,C) are maps, (a,b,c) are variables, and H is a constant (problem data)
	Primal constraints are equality constraits (Aa+Bb==0), dual constraints are inequalities (>=0).
	"""
	
	primal_constraints = []
	dual_constraints = []
	
	def __init__(self, label, sign_list, map_list, var_list, primal_or_dual, constr_type, constant = None, conjugateVar = None):
		
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
			assert np.prod(conjugateVar.dims) == self.maps[0].dims['out'] , f'Dimension mismatch between constraint {label} and dual variable {conjugateVar.name}'
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
		
		constr_list.append(self)
		
		
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
		
		
	'''
	if Maps.check_inputs:
			assert (in_var.primal_or_dual == 'primal' and out_var.primal_or_dual == 'dual') or \
					(in_var.primal_or_dual == 'dual' and out_var.primal_or_dual == 'primal'),  \
					('in_var and out_var sould be a prima-dual pair\n.' + f'in_var {in_var.name} is {in_var.primal_or_dual}\n' + \
					f'out_var {out_var.name} in {out_var.primal_or_dual}' ) 
				
		
	'''
	

def print_constraint(c):
		"""
		print the expression for the constraint
		"""
		print(f"{c.label:12} : {c.conjugateVar.name:12} : {c.constr_type:12} : ", end=' ')
		
		for (sign, CGmap, optVar) in zip(c.signs, c.maps, c.var_list):
			print(f"{signString(sign)} {CGmap.name}({optVar.name}) ", end = '')
		
		if c.constant:
			print(c.constant, end = ' ')
		
		print('\n')

def signString(x): 
	s = f"{x:+}"
	if abs(x) == 1:
		s = s[0]
	return s
