
import maps
from variables import OptVar
from variables import RestrictedKeysDict
from constraints import Constraint
import matrix_aux_functions as mf
import numpy as np
import argparse
from scs_funcs import SCS_Solver


"""
	********** problem formulation  
	as in https://arxiv.org/abs/2212.03014
	********************************************
	primal:						| conj Var
	min tr(rho*H)
	st 	rho >= 0 				: 
		tr(rho) == 1			: epsilon
		tr_L(rho) == tr_R(rho)	: a
	
	dual:									| conjVar
	max delta
	st	H - epsilon*\id + \id\otimes a - a\otimes\id >= 0		: rho 
		
	
	
	*********** scs convention: 
	as in https://doi.org/10.1007/s10957-016-0892-3
	**********************************************
	dual:
		max -b.T*y
		st -A.T*y = c
		y >= 0
	
	primal:
		min c.T*x
		st A*x + s = b
		s >= 0
		x \in R^n
		 
	
	==========> we read primal problem as the scs dual:
	b = H
	y = rho
	A.T(rho) = [ -tr(rho*H), -tr_L(rho) + tr_R(rho)]
	c =  [1,0]
	
	
	"""


exact_sol = 0.25 -np.log(2) # exact sol heisenberg
 
 
 
 

def define_arguments():
	'''
	define the command line arguments required to define this problem
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("n", help="system size", type=int)
	
	return parser
	

 

def set_problem_and_make_solver(args, settings):
	
	print('>>>>>>>>>>>>>>>>>>> PROBLEM:  LTI N <<<<<<<<<<<<<<<<<<<<<<<<<<')
	
	n = args.n
	d=2 
	assert n>2, 'n must be > 2' 
	
	dims_rho = (d,)*(n-1)
	dims_omega = (d,)*(n)

	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)
	omega = OptVar('omega','primal', dims = dims_omega, cone = 'PSD', dtype = float)
	
	rng = np.random.default_rng(seed=166)
	
	# heisenberg model with sub-lattice rotation -H_xxz(1,1,-1)
	h_term = np.array(  [[0.25, 0, 0, 0],
					[0, -0.25, -0.5, 0],
					[0, -0.5, -0.25, 0],
					[0, 0, 0, 0.25]]
				)
	
	H = np.kron(h_term, np.identity(d**(n-1-2))) # extend to size of rho
	
	
	tr_l_omega = maps.PartTrace(subsystems = [1], state_dims = dims_omega )
	tr_r_omega = maps.PartTrace(subsystems = [n], state_dims = dims_omega )

	tr = maps.Trace(dim = rho.matdim )

	id_rho = maps.Identity( dim = rho.matdim)
	# id_omega = maps.Identity( dim = states[k0+2].matdim)
	# id_1 = maps.Identity( dim = 1)



	# dual varsiables
	# a = OptVar('alpha', 'dual', dims = (d,)*(n-2) , dtype = float )
	b_l = OptVar('beta_l', 'dual', dims = dims_rho, dtype = float)
	b_r = OptVar('beta_r', 'dual', dims = dims_rho, dtype = float)
	 
	e = OptVar('epsilon', 'dual', dims = (1,), dtype = float)
	
	
	
	# primal constraints
	# for the sign conventions see sign_convention_doc.txt
	# the LHS in the primal constraints is -A.T
	# the RHS (constants) are without a minus sign,
	# i.e., for a constraint tr(rho)==1 we need to put -tr(rho) in the primal constraints and const = 1
	
	Constraint(**{
		'label': 'norm', 
		'sign_list': [-1,],     # -tr(rho)
		'map_list': [tr,],
		'adj_flag_list': [False,],
		'var_list': [rho,],
		'primal_or_dual': 'primal',
		'conjugateVar': e,
		'const' : 1.0       
		})
	
	# when no constant is specified (or is None)
	# the RHS is set as a vector of zeros of the correct size:
	# np.zeros((conjugateVar.matdim, conjugateVar.matdim))
	
	
	
	# Constraint(**{
		# 'label': 'LTI', 
		# 'sign_list': [ +1,],
		# 'map_list': [tr_l_rho_MINUS_tr_r_rho],
		# 'adj_flag_list': [False, ],
		# 'var_list': [rho,],
		# 'primal_or_dual': 'primal',
		# 'conjugateVar': a,
		# })
		
	Constraint(**{
		'label': 'left', 
		'sign_list': [-1, +1],
		'map_list': [id_rho, tr_l_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, omega],
		'primal_or_dual': 'primal',
		'conjugateVar': b_l,
		})
	
	Constraint(**{
		'label': 'right', 
		'sign_list': [-1, +1],
		'map_list': [id_rho, tr_r_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, omega],
		'primal_or_dual': 'primal',
		'conjugateVar': b_r,
		})
	
	
	# dual constraints
	
	# Constraint(**{
			# 'label': 'D_rho', 
			# 'sign_list':  [ -1, -1, +1, -1],
			# 'map_list': [id_rho, id_rho, tr_l_rho_MINUS_tr_r_rho, tr ],
			# 'adj_flag_list': [True, True, True,  True ],
			# 'var_list': [ b_l, b_r, a, e],
			# 'primal_or_dual': 'dual',
			# 'conjugateVar': rho,
			# 'const': H.ravel()
			# }) 
	Constraint(**{
		'label': 'D_rho', 
		'sign_list':  [ -1, -1, -1],
		'map_list': [id_rho, id_rho,  tr ],
		'adj_flag_list': [True, True,  True ],
		'var_list': [ b_l, b_r, e],
		'primal_or_dual': 'dual',
		'conjugateVar': rho,
		'const': H.ravel()
	}) 
	  
	Constraint(**{
			'label': "D_omega", 
			'sign_list':  [+1, +1,],
			'map_list': [tr_l_omega, tr_r_omega  ]  ,
			'adj_flag_list': [True, True ],
			'var_list': [b_l, b_r],
			'primal_or_dual': 'dual',
			'conjugateVar': omega,
			}) 

	 
	Constraint.print_constr_list()
	
	OptVar._close_var_lists()
	
	solver = SCS_Solver(settings, exact_sol = exact_sol)
	
	return solver
