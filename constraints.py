import CGmaps
from variables import OptVar 

class Constraint:
	""" Each constraint is an expression of the form
	A*a + B*b + C*c +... + H 
	where (A,B,C) are maps, (a,b,c) are variables, and H is a constant (problem data)
	Primal constraints are equality constraits (Aa+Bb==0), dual constraints are inequalities (>=0).
	"""
	
	all_constraints = []
	
	def __init__(self, label, maps, optVars, constant = None, dualVar = None):
		
		# todo exception if len(maps) != len(optVars)
		self.label = label # constraint name
		self.maps = maps
		self.optVars = optVars
				
		if dualVar: 
			self.dualVar = dualVar
		else:
			self.dualVar = OptVar(f'MISSING VAR:{label}:','dual')
		
		self.constant = constant
		
		Constraint.all_constraints.append(self)
		
	def print_constr_list(self):
		for c in Constraint.all_constraints:
			print_constraint(c)
		
		
	
	

def print_constraint(c):
		"""
		print the expression for the constraint
		"""
		print(f"{c.label:16} :: {c.dualVar.name:16}: ", end=' ')
		
		for (CGmap, optVar) in zip(c.maps, c.optVars):
			print(f"{signString(CGmap.sign)} {CGmap.name}({optVar.name}) ", end = '')
		
		if c.constant:
			print(c.constant)
		
		print('\n')

def signString(x): 
	s = f"{x:+}"
	if abs(x) == 1:
		s = s[0]
	return s
