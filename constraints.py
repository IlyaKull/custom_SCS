import CGmaps
from variables import OptVar 
import numpy as np

class Constraint:
	""" Each constraint is an expression of the form
	A*a + B*b + C*c +... + H 
	where (A,B,C) are maps, (a,b,c) are variables, and H is a constant (problem data)
	Primal constraints are equality constraits (Aa+Bb==0), dual constraints are inequalities (>=0).
	"""
	
	all_constraints = []
	
	def __init__(self, label, maps, optVars, constant = None, dualVar = None):
		
		assert len(maps) == len(optVars), f'Supplied Maps and variables do not match in length in constraint {label}'
		self.label = label # constraint name
		self.maps = maps
		self.optVars = optVars
				
		try:
			assert all( m.dims['out'] == maps[0].dims['out'] for m in maps ), \
				f'Output dimension mismatch in constraint "{label}"'
		except AssertionError as xxx:
			print('!!!!!!!!!!!!! output dims:')
			for m in maps:
				print(m.dims['out'])
			raise xxx
		else:
			if dualVar: 
				self.dualVar = dualVar
				assert np.prod(dualVar.dims) == maps[0].dims['out'] , f'Dimension mismatch between constraint {label} and dual variable {dualVar.name}'
			else:
				self.dualVar = OptVar(f'MISSING VAR:{label}:','dual', dims = {'totaDims': maps[0].dims['out']})
			
		self.constant = constant
		
		
		Constraint.all_constraints.append(self)
		
		
	def print_constr_list(self):
		print('~'*30 + ' CONSTRAINTS: '.center(20) + '~'*30 )
		print(f'Total of {len(Constraint.all_constraints)} constraints')
		print('='*80)
		print('name'.ljust(16) +' : ' + 'dual var'.ljust(16) + ' : expression') 
		print('-'*80)
		for c in Constraint.all_constraints:
			print_constraint(c)
		print('='*80)
		
	
	

def print_constraint(c):
		"""
		print the expression for the constraint
		"""
		print(f"{c.label:16} : {c.dualVar.name:16} : ", end=' ')
		
		for (CGmap, optVar) in zip(c.maps, c.optVars):
			print(f"{signString(CGmap.sign)} {CGmap.name}({optVar.name}) ", end = '')
		
		if c.constant:
			print(c.constant, end = ' ')
		
		print('\n')

def signString(x): 
	s = f"{x:+}"
	if abs(x) == 1:
		s = s[0]
	return s
