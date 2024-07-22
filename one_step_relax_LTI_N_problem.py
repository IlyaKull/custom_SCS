
import maps
from variables import OptVar
from variables import RestrictedKeysDict
from constraints import Constraint
import matrix_aux_functions as mf
import numpy as np
import util

exact_sol = 0.25 -np.log(2) # exact sol heisenberg

		

def set_problem(chi , d, xOtimesI_impl = 'kron', cg_impl = 'kron'):
	
	print('>>>>>>>>>>>>>>>>>>> PROBLEM:  ONE STEP RELAX LTI N <<<<<<<<<<<<<<<<<<<<<<<<<<')
	
	k0 = util.determine_k0(d,chi)
	print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>> k0 = {k0}')
	n = k0+2 
	print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>> n = {n}')
	
	
	dims_rho = (d,)*(n-1)
	dims_omega = (d, chi, d)

	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)
	omega = OptVar('omega','primal', dims = dims_omega , cone = 'PSD', dtype = float)
	
	rng = np.random.default_rng(seed=166)
	
	# heisenberg model with sub-lattice rotation -H_xxz(1,1,-1)
	h_term = np.array(  [[0.25, 0, 0, 0],
					[0, -0.25, -0.5, 0],
					[0, -0.5, -0.25, 0],
					[0, 0, 0, 0.25]]
				)
	
	H = np.kron(h_term, np.identity(d**(n-1-2))) # extend to size of rho
	
	tr_l_rho = maps.PartTrace(subsystems = [1], state_dims = dims_rho, implementation = xOtimesI_impl )
	tr_r_rho = maps.PartTrace(subsystems = [n-1], state_dims = dims_rho, implementation = xOtimesI_impl )
	
	tr_l_rho_MINUS_tr_r_rho = maps.AddMaps([tr_l_rho, tr_r_rho], [-1, +1])
			
	tr_l_omega = maps.PartTrace(subsystems = [1], state_dims = dims_omega, implementation = xOtimesI_impl )
	tr_r_omega = maps.PartTrace(subsystems = [3], state_dims = dims_omega, implementation = xOtimesI_impl )

	tr = maps.Trace(dim = rho.matdim )

	id_rho = maps.Identity( dim = rho.matdim)
	# id_omega = maps.Identity( dim = states[k0+2].matdim)
	# id_1 = maps.Identity( dim = 1)
	
	
	# cg maps acting on rho
	action_l = {'dims_in': dims_rho, 'pattern':(1,)*k0 + (0,), 'pattern_adj':(1,0), 'dims_out':(chi ,d)}
	action_r = {'dims_in': dims_rho, 'pattern':(0,) + (1,)*k0, 'pattern_adj':(0,1), 'dims_out':(d, chi)}
	
	rand_map = rng.random((chi ,d**k0))
	isometry = np.linalg.qr(rand_map.T)[0]
	print('<<<<<<<<<<<<<<<<<<<<<<<<', isometry.shape)
	krausOps = [isometry.T, ]
	C_l = maps.CGmap('C_l', krausOps, action = action_l , implementation = cg_impl)
	C_r = maps.CGmap('C_r', krausOps, action = action_r , implementation = cg_impl)


	# dual varsiables
	a = OptVar('alpha', 'dual', dims = (d,)*(n-2) , dtype = float )
	b_l = OptVar('beta_l', 'dual', dims = (chi, d), dtype = float)
	b_r = OptVar('beta_r', 'dual', dims = (d, chi), dtype = float)
	 
	e = OptVar('epsilon', 'dual', dims = (1,), dtype = float)
	
	
	
	# primal constraints
	
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
	
	Constraint(**{
		'label': 'LTI', 
		'sign_list': [ -1,],
		'map_list': [tr_l_rho_MINUS_tr_r_rho],
		'adj_flag_list': [False, ],
		'var_list': [rho,],
		'primal_or_dual': 'primal',
		'conjugateVar': a,
		})
		
	Constraint(**{
		'label': 'cg_left', 
		'sign_list': [-1, +1],
		'map_list': [C_l, tr_l_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, omega],
		'primal_or_dual': 'primal',
		'conjugateVar': b_l,
		})
	
	
	Constraint(**{
		'label': 'cg_right', 
		'sign_list': [-1, +1],
		'map_list': [C_r, tr_r_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, omega],
		'primal_or_dual': 'primal',
		'conjugateVar': b_r,
		})
	
	
	# dual constraints 
	Constraint(**{
			'label': 'D_rho', 
			'sign_list':  [ -1, -1, -1, -1],
			'map_list': [C_l, C_r, tr_l_rho_MINUS_tr_r_rho, tr ],
			'adj_flag_list': [True, True, True,  True ],
			'var_list': [ b_l, b_r, a, e],
			'primal_or_dual': 'dual',
			'conjugateVar': rho,
			'const': H.ravel()
			}) 
			

	Constraint(**{
			'label': 'D_omega', 
			'sign_list':  [+1, +1,],
			'map_list': [tr_l_omega, tr_r_omega  ]  ,
			'adj_flag_list': [True, True ],
			'var_list': [b_l, b_r],
			'primal_or_dual': 'dual',
			'conjugateVar': omega,
			}) 
		 
	Constraint.print_constr_list()
	
	OptVar._close_var_lists()
	
	return 0
