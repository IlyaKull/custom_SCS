
import maps
from variables import OptVar
from variables import RestrictedKeysDict
from constraints import Constraint
import matrix_aux_functions as mf
import numpy as np
import util
import CGmaps  
import logging
from scs_funcs import SCS_Solver
logger = logging.getLogger(__name__)

import argparse

"""
	********** problem formulation 
	as in https://arxiv.org/abs/2212.03014
	*************************************
	primal:	(**)								| conj Var
	min tr(rho_k0*H)
	st 	rho_k0 >= 0 							: 
		omega_k >= 0  for all k=k0+1...n		: 
		tr(rho_k0) == 1							: epsilon
		tr_L(rho_k0) == tr_R(rho_k0)			: a
		V0_L(rho_k0) == tr_L(omega_{k0+1})		: b_L
		V0_R(rho_k0) == tr_R(omega_{k0+1})		: b_R
		L_{k0+1}(omega_{k0+1}) == tr_L(omega_{k0+2})	: g^{k0+1}_L
		R_{k0+1}(omega_{k0+1}) == tr_R(omega_{k0+2})	: g^{k0+1}_R
		and so on until
		L_{n-1}(omega_{n-1}) == tr_L(omega_{n})			: g^{n-1}_L
		R_{n-1}(omega_{n-1}) == tr_R(omega_{n})			: g^{n-1}_R
		
	dual:																							| conjVar
	max delta
	st	H - epsilon*I + Ixa - axI + V0_L^dag(b_L) + V0_R^dag(b_R) >= 0								: rho_k0
		Ixb_L + b_RxI - L_{k0+1}^dag(g^{k0+1}_L) - R_{k0+1}^dag(g^{k0+1}_R) >= 0					: omega_{k0+1}
		Ix g^{k0+1}_L + g^{k0+1}_L xI - L_{k0+2}^dag(g^{k0+2}_L) - R_{k0+2}^dag(g^{k0+2}_R) >= 0	: omega_{k0+2}
		and so on 
		Ix g^{k0+1}_L + g^{k0+1}_L xI  >= 0															: omega_{n}
	
	
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
		 
	
	==========> we read problem into scs dual:
	b = [H,0,0,...]^T
	y = [rho_k0, omega_{k0+1}, omega_{k0+1},..., omega_{n}]^T 
	A.T*y = [ -tr(rho*H), -tr_L(rho) + tr_R(rho), V0_L(rho_k0) == tr_L(omega_{k0+1}), V0_R(rho_k0) == tr_R(omega_{k0+1}),... and so on the constraints in he primal (**) above  ]
	c =  [1,0,0,0.....]^T
	
	
	"""



def define_arguments():
	'''
	define the command line arguments required to define this problem
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("n", help="system size", type=int)
	parser.add_argument("D", help="bond dim of MPS used for compression", type=int)
	parser.add_argument("--mps_filename", help="file where compression mps is saved", type=str, default = '')
	parser.add_argument("--coarse_grain_method", help="which coarse graining maps to construct from given MPS", 
					type=str, default = "plain", choices=["plain", "isometries"])

	return parser
	


def set_problem_and_make_solver(args, settings):

	logger.info('PROBLEM:  RELAX LTI N')
	
	n = args.n
	D = args.D
	d=2
	
	if len(args.mps_filename)>0:
		mps = read_mps_from_file(args.mps_filename)
		try:
			assert mps.shape == (D,d,D)
		except AssertionError:
			logger.critical(f'loaded mps dimensions {mps.shape} are incompatible with problem dimensions {(D,d,D)}')
			raise
	else:
		logger.critical("no MPS file specified!!! proceeding with random MPS")
		rng = settings['util_rng']
		mps = rng.random((D,d,D,)) 
		
	
	cg_impl = 'kron' #which way to implement cg maps. see maps.py>CGmaps...implementation
		 
	
	
	
	# heisenberg model with sub-lattice rotation -H_xxz(1,1,-1)
	h_term = np.array(  [[0.25, 0, 0, 0],
					[0, -0.25, -0.5, 0],
					[0, -0.5, -0.25, 0],
					[0, 0, 0, 0.25]]
				)
	exact_sol = 0.25 -np.log(2) # exact sol heisenberg
	
	k0 = util.determine_k0(d, chi = D**2)
	H = np.kron(h_term, np.identity(d**(k0+1-2))) # extend to size of rho
	
	
	
	
	logger.info(f'k0 = {k0}, base state: rho_{k0+1}, final state omega_{n}')
	
	assert n >= k0+2 , f'n has to be at least {k0+2}: n = {n}'
	 
	
	
	dims_rho = (d,)*(k0+1)
	dims_omega = (d,D,D,d)

	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)
	
	
	
	
	# states is a dict with >>key = the number of spins<<
	states = RestrictedKeysDict(allowed_keys = list(range(k0+1,n+1)))
	states[k0+1] = rho
	for k in range(k0+2,n+1):
		states[k] = OptVar(f"omega_{k}",'primal', dims = dims_omega, cone = 'PSD', dtype = float)

	
	# compute isometries for CG
	if args.coarse_grain_method == "isometries":
		V0, L, R = CGmaps.iso_from_MPS(mps, k0, n)
	else:
		V0, L, R = CGmaps.plain_cg_from_MPS(mps, k0, n)

	
	
	# cg maps acting on rho
	action_l0 = {'dims_in': dims_rho, 'pattern':(1,)*k0 + (0,), 'pattern_adj':(1,1,0), 'dims_out':(D,D,d)}
	action_r0 = {'dims_in': dims_rho, 'pattern':(0,) + (1,)*k0, 'pattern_adj':(0,1,1), 'dims_out':(d,D,D)}
	krausOps0 = [V0, ]
	C_l0 = maps.CGmap('C_l0', krausOps0, action = action_l0 , implementation = cg_impl)
	C_r0 = maps.CGmap('C_r0', krausOps0, action = action_r0 , implementation = cg_impl)

	# cg maps acting on omegas 
	action_l1 = {'dims_in': dims_omega, 'pattern':(1,1,1,0), 'pattern_adj':(1,1,0), 'dims_out':(D,D,d)}
	action_r1 = {'dims_in': dims_omega, 'pattern':(0,1,1,1), 'pattern_adj':(0,1,1), 'dims_out':(d,D,D)}
	
	C_l = [[],]*(n+1)
	C_r = [[],]*(n+1)
	for k in range(k0+2,n):
		C_l[k] = maps.CGmap(f'C_l{k}', [L[k-2],], action = action_l1 , implementation = cg_impl)
		C_r[k] = maps.CGmap(f'C_r{k}', [R[k-2],], action = action_r1 , implementation = cg_impl)

	
	
	tr_l_rho = maps.PartTrace(subsystems = [1], state_dims = dims_rho )
	tr_r_rho = maps.PartTrace(subsystems = [k0+1], state_dims = dims_rho )
	
	tr_l_rho_MINUS_tr_r_rho = maps.AddMaps([tr_l_rho, tr_r_rho], [-1, +1])
			
	tr_l_omega = maps.PartTrace(subsystems = [1], state_dims = dims_omega )
	tr_r_omega = maps.PartTrace(subsystems = [4], state_dims = dims_omega )

	tr = maps.Trace(dim = rho.matdim )

	# id_rho = maps.Identity( dim = rho.matdim)
	# id_omega = maps.Identity( dim = states[k0+2].matdim)
	# id_1 = maps.Identity( dim = 1)
	
	# dual varsiables
	a = OptVar('alpha', 'dual', dims = (d,)*k0 , dtype = float )
	b_l = OptVar('beta_l', 'dual', dims = (D,D,d), dtype = float)
	b_r = OptVar('beta_r', 'dual', dims = (d,D,D), dtype = float)
	g_l = RestrictedKeysDict(allowed_keys = list(range(k0+1,n))) 
	g_r = RestrictedKeysDict(allowed_keys = list(range(k0+1,n)))

	# beta_l = g_l[k0+1] is dual to V*rho*V^+ == pTr_l omega_k0+2
	# g_l[k0+2] is dual to V*omega_k0+2*V^+ == pTr omega_k0+3 and so on
	g_l[k0+1] = b_l
	g_r[k0+1] = b_r

	for k in range(k0+2,n):
		g_l[k] = OptVar(f"gamma_L_{k}", 'dual', dims = (D,D,d), dtype = float)
		g_r[k] = OptVar(f"gamma_R_{k}", 'dual', dims = (d,D,D), dtype = float)

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
		'const' : 1.0,
		'const_name' : '1'      
		})
	
	# when no constant is specified (or is None)
	# the RHS is set as a vector of zeros of the correct size:
	# np.zeros((conjugateVar.matdim, conjugateVar.matdim))
	
	
	
	Constraint(**{
		'label': 'LTI', 
		'sign_list': [ +1,],
		'map_list': [tr_l_rho_MINUS_tr_r_rho],
		'adj_flag_list': [False, ],
		'var_list': [rho,],
		'primal_or_dual': 'primal',
		'conjugateVar': a,
		})
		
	Constraint(**{
		'label': f'left_{k0+1}', 
		'sign_list': [-1, +1],
		'map_list': [C_l0, tr_l_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, states[k0+2]],
		'primal_or_dual': 'primal',
		'conjugateVar': b_l,
		})
	
	
	Constraint(**{
		'label': f'right_{k0+1}', 
		'sign_list': [-1, +1],
		'map_list': [C_r0, tr_r_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, states[k0+2]],
		'primal_or_dual': 'primal',
		'conjugateVar': b_r,
		})
	
	for k in range(k0+2,n):
	
		Constraint(**{
			'label': f'left_{k}', 
			'sign_list': [-1, +1],
			'map_list': [C_l[k], tr_l_omega],
			'adj_flag_list': [False, False],
			'var_list': [states[k], states[k+1]],
			'primal_or_dual': 'primal',
			'conjugateVar': g_l[k],
			})
			
		Constraint(**{
			'label': f'right_{k}', 
			'sign_list': [-1, +1],
			'map_list': [C_r[k], tr_r_omega],
			'adj_flag_list': [False, False],
			'var_list': [states[k], states[k+1]],
			'primal_or_dual': 'primal',
			'conjugateVar': g_r[k],
			})
		
		
	
	# dual constraints
	
	Constraint(**{
			'label': f'D_{k0+1}', 
			'sign_list':  [ -1, -1, +1, -1],
			'map_list': [C_l0, C_r0, tr_l_rho_MINUS_tr_r_rho, tr ],
			'adj_flag_list': [True, True, True,  True ],
			'var_list': [ b_l, b_r, a, e],
			'primal_or_dual': 'dual',
			'conjugateVar': rho,
			'const': H.ravel(),
			'const_name' : 'H'
			}) 
			
	for k in range(k0+2,n):
		# RECALL:
		# beta_l = g_l[k0+1] is dual to V*rho*V^+ == pTr_l omega_k0+2
	
		Constraint(**{
				'label': f"D_{k}", 
				'sign_list':  [+1, +1, -1, -1],
				'map_list': [tr_l_omega, tr_r_omega , C_l[k], C_r[k] ]  ,
				'adj_flag_list': [True, True, True, True ],
				'var_list': [g_l[k-1], g_r[k-1], g_l[k], g_r[k]],
				'primal_or_dual': 'dual',
				'conjugateVar': states[k],
				}) 

	Constraint(**{
			'label': f"D_{n}", 
			'sign_list':  [+1, +1,],
			'map_list': [tr_l_omega, tr_r_omega  ]  ,
			'adj_flag_list': [True, True ],
			'var_list': [g_l[n-1], g_r[n-1]],
			'primal_or_dual': 'dual',
			'conjugateVar': states[n],
			}) 

	if logging.DEBUG >= logging.root.level:
		Constraint.print_constr_list()
	
	OptVar._close_var_lists()
	
	solver = SCS_Solver(settings, exact_sol = exact_sol)
	
	return solver
	


def read_mps_from_file(mps_filename):
	try:
		with open(mps_filename, 'rb') as f:
			mps = np.load(f)
	except FileNotFoundError:
		logger.critical(f"specified file '{mps_filename}' not found")
		raise
		
	return mps
	
