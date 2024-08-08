import numpy as np
import CGmaps

from variables import OptVar 

import maps
from util import sign_str

import matrix_aux_functions as mf

import logging
logger = logging.getLogger(__name__)


class Constraint:
	""" Each constraint is an expression of the form
	A*a + B*b + C*c +... + H 
	where (A,B,C) are maps, (a,b,c) are variables, and H is a constant (problem data)
	Primal constraints are equality constraits (Aa+Bb==0), dual constraints are inequalities (>=0).
	"""
	
	primal_constraints = []
	dual_constraints = []
	maps_table = dict()
	
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
 		
		if const is None:
			self.const = np.zeros(conjugateVar.matdim**2)
		else:
			self.const = const
			
				
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
		
		logger.debug(f'adding constraint {self.label} to {primal_or_dual} constraint list')
		if add_to_constr_list:
			constr_list.append(self)
			
		self._add_maps_to_table()
		
		
	
	def _add_maps_to_table(self):
		'''
		each added constraint adds its maps to a 'maps table' which is labeled by 
		(dual var, pimal var). 
		The dual constraints should be the adoint of the primal ones, 
		i.e. should result in the transposed of the 'maps table' of the primal constraints.
		This condition is checked when each constraint is added:
		primal constraints are added to the table, dual constraints are checked against the primal ones.		
		'''
		
		if self.primal_or_dual == 'primal':
			for s,m,f,v in zip(self.signs, self.maps, self.adj_flags, self.var_list):
				
				if (self.conjugateVar, v) in Constraint.maps_table:
					raise Exception(f"when addig primal constraint {self.label} maps table entry {(self.conjugateVar.name, v.name)} already occupied")
				else:
					Constraint.maps_table[(self.conjugateVar, v)] = {'sign' : s,\
																	'map' : m,\
																	'adj_flag' : f,\
																	'ticked_by_dual' : False\
																	}
		if self.primal_or_dual == 'dual':
			for s,m,f,v in zip(self.signs, self.maps, self.adj_flags, self.var_list):
					# print(f"pair {v.name, self.conjugateVar.name}")
					if (v, self.conjugateVar) in Constraint.maps_table:
						# print('found in table')
						ts = Constraint.maps_table[(v, self.conjugateVar)]['sign']
						tm = Constraint.maps_table[(v, self.conjugateVar)]['map']
						tf = Constraint.maps_table[(v, self.conjugateVar)]['adj_flag']
						tt = Constraint.maps_table[(v, self.conjugateVar)]['ticked_by_dual']
						assert ts == s and tm == m and tf != f and tt == False,\
							(f"when adding dual constraint {self.label} maps table entry {(v.name, self.conjugateVar.name)} doesn't match currnt input\n",\
							f"table entry: {ts, tm.name, tf}; current input: {s, m.name, f} (falgs match if opposite)")
						Constraint.maps_table[(v, self.conjugateVar)]['ticked_by_dual'] = True
					else:
						raise Exception(f"when adding dual constraint {self.label} maps table entry  {(v.name, self.conjugateVar.name)} was not found.\n ADD ALL PRIMAL CONSTRAINTS FIRST")
					
		
		
		
	# @profile
	def __call__(self, v_in, v_out):
		"""
		Each constraint is an expression of the form \sum_i M_i(v_i) for some maps M_i and variables v_i.
		To each constraint ther is an associated conjugate variable. 
		When a (primal or dual) constraint is called, the expression \sum_i M_i(v_i) is added to the the conjugate var
		I.E. CONSTRAINTS ALWAYS ACT IN PLACE  as +=
		"""
		
		if logging.root.level==0:
			logger.notset(f"calling constraint {self.label}")
		
		if not self.conjugateVar.added_to_var_list:
			print('~~~~~~~~~~ !!!!!!!!!!!!!!!!! \n',\
			f'the variable conjugate to constraint {self.label} ({self.conjugateVar.name})', \
			'was not added to the list of variables and therefore does not have a buffer assigned' )
			return 1
		
		 
		if logging.root.level==0:
			logger.notset(f"conj var: {self.conjugateVar.name}")
		
		for s,M,f,var in zip(self.signs, self.maps, self.adj_flags, self.var_list):
			if logging.root.level==0:
				logger.notset(f"s = {s}")
				logger.notset(f"M = {M.name}")
				logger.notset(f"adj = {f}")
				logger.notset(f"var = {var}")
				logger.notset(f"out var slice = {v_out[ self.conjugateVar.slice ].shape}")
				 
			
			 
			# maps act in place as += 
			M.__call__( var, v_in, sign = s, adj_flag = f,  out = v_out[ self.conjugateVar.slice ]) 
			
			
			
		return 0
	
	@classmethod
	def print_constr_list(cls):
		width = 15
		for i, constr_list in enumerate((cls.primal_constraints, cls.dual_constraints)):
			print('='*100)
			print('{} CONSTRAINTS: '.format( ('PRIMAL', 'DUAL')[i]).center(100)  )
			print(f'Total of {len(constr_list)} constraints')
			print('='*100)
			print('name'.ljust(width) +' : ' + 'conj var'.ljust(width) + ' : expression') 
			print('-'*100)
			
			for c in constr_list:
				cls.print_constraint(c, width)
	
	
	@classmethod
	def _check_maps_table(cls):
		'''
		all expressions that appear in the primal also appear in the dual. This is checked by the Constraint.maps_table variable. 
		As primal constraints are added they insert maps into the entries of the maps_table which is labeled by (dual vars, primal vars). 
		Since the dual constraints have the adjoint constraint matrix, when a dual constr is added
		it is checked that it hits an existing entry in the table. This is done by the Constraint._add_maps_to_table() method and ensures that 
		(dual const mat) \subset (primal constr mat)^T.
		Here we check that every primal constraint was then visited by a dual constraint (the 'ticked_by_dual' field). which completes the check
		A^T.conj() shold be the adjoint of A.
		'''
		
		try:
			assert all( [v['ticked_by_dual'] for v in cls.maps_table.values()]), 'dual constraints missing!'
		except AssertionError:
			print('pairs of variables which appear in primal but not in dual have value False:')
			print( {(k[0].name, k[1].name): v['ticked_by_dual'] for k,v in cls.maps_table.items()} )
			raise
		else:
			logger.info('Maps table checked')
	
	@staticmethod
	def print_constraint(c, width):
			"""
			print the expression for the constraint
			"""
			print(c.label.ljust(width) +  " : " + c.conjugateVar.name.ljust(width) +  " : ", end=' ')
			
			for (sign, CGmap, adj_flag, optVar) in zip(c.signs, c.maps, c.adj_flags, c.var_list):
				if adj_flag:
					print(f"{sign_str(sign)} {CGmap.adj_name}({optVar.name}) ", end = '')
				else:
					print(f"{sign_str(sign)} {CGmap.name}({optVar.name}) ", end = '')
			
			print('\n')

	 
		
		
		
		
		
		
		
		
