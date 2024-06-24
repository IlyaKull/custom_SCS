
import maps
from variables import OptVar
from variables import RestrictedKeysDict
from constraints import Constraint, print_constraint
import matrix_aux_functions as mf
import numpy as np


def determine_k0(d,D):
	
	try:
		k0 = next( k for k in range(100) if D**2 < d**k)	
	except StopIteration:
		print('Could not determine k0 value < 100' )
		raise
	
	return k0
		

def set_problem(n,D,d, xOtimesI_impl = 'kron', cg_impl = 'kron'):

	rng = np.random.default_rng(seed=17)
	
	 
	
	k0 = determine_k0(d,D)
	print(f'k0={k0}')
	
	assert n >= k0+2 , f'n has to be at least {k0+2}: n = {n}'
	 
	
	
	dims_rho = (d,)*(k0+1)
	dims_omega = (d,D,D,d)

	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)

	# states is a dict with >>key = the number of spins<<
	states = RestrictedKeysDict(allowed_keys = list(range(k0+1,n+1)))
	states[k0+1] = rho
	for k in range(k0+2,n+1):
		states[k] = OptVar(f"omega_{k}",'primal', dims = dims_omega, cone = 'PSD', dtype = float)

		
	# cg maps acting on rho
	action_l0 = {'dims_in': dims_rho, 'pattern':(1,)*k0 + (0,), 'pattern_adj':(1,1,0), 'dims_out':(D,D,d)}
	action_r0 = {'dims_in': dims_rho, 'pattern':(0,) + (1,)*k0, 'pattern_adj':(0,1,1), 'dims_out':(d,D,D)}
	krausOps0 = [rng.random((D**2,d**k0)), ]
	C_l0 = maps.CGmap('C_l0', krausOps0, action = action_l0 , implementation = cg_impl)
	C_r0 = maps.CGmap('C_r0', krausOps0, action = action_r0 , implementation = cg_impl)

	# cg maps acting on omegas 
	action_l1 = {'dims_in': dims_omega, 'pattern':(1,1,1,0), 'pattern_adj':(1,1,0), 'dims_out':(D,D,d)}
	action_r1 = {'dims_in': dims_omega, 'pattern':(0,1,1,1), 'pattern_adj':(0,1,1), 'dims_out':(d,D,D)}
	krausOps1 = [rng.random((D**2,D*D*d)),]
	C_l1 = maps.CGmap('C_l1', krausOps1, action = action_l1 , implementation = cg_impl)
	C_r1 = maps.CGmap('C_r1', krausOps1, action = action_r1 , implementation = cg_impl)

	
	
	tr_l_rho = maps.PartTrace(subsystems = [1], state_dims = dims_rho, implementation = xOtimesI_impl )
	tr_r_rho = maps.PartTrace(subsystems = [k0+1], state_dims = dims_rho, implementation = xOtimesI_impl )
			
	tr_l_omega = maps.PartTrace(subsystems = [1], state_dims = dims_omega, implementation = xOtimesI_impl )
	tr_r_omega = maps.PartTrace(subsystems = [4], state_dims = dims_omega, implementation = xOtimesI_impl )

	tr = maps.Trace(dim = rho.matdim )

	id_rho = maps.Identity( dim = rho.matdim)
	id_omega = maps.Identity( dim = states[k0+2].matdim)
	id_1 = maps.Identity( dim = 1)

	H_map = maps.TraceWith( 'H', operator = np.identity(rho.matdim, dtype=float) ,dim = rho.matdim )

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
	 
	Constraint(**{
		'label': 'norm', 
		'sign_list': [-1,],
		'map_list': [tr,],
		'adj_flag_list': [False,],
		'var_list': [rho,],
		'primal_or_dual': 'primal',
		'conjugateVar': e,
		})
	
	Constraint(**{
		'label': 'LTI', 
		'sign_list': [-1, +1],
		'map_list': [tr_l_rho, tr_r_rho],
		'adj_flag_list': [False, False],
		'var_list': [rho, rho],
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
			'map_list': [C_l1, tr_l_omega],
			'adj_flag_list': [False, False],
			'var_list': [states[k], states[k+1]],
			'primal_or_dual': 'primal',
			'conjugateVar': g_l[k],
			})
			
		Constraint(**{
			'label': f'right_{k}', 
			'sign_list': [-1, +1],
			'map_list': [C_r1, tr_r_omega],
			'adj_flag_list': [False, False],
			'var_list': [states[k], states[k+1]],
			'primal_or_dual': 'primal',
			'conjugateVar': g_r[k],
			})
		
		
	
	# dual constraints
	
	Constraint(**{
			'label': f'D_{k0+1}', 
			'sign_list':  [ -1, -1, -1, +1, -1],
			'map_list': [C_l0, C_r0, tr_l_rho, tr_r_rho, id_rho ],
			'adj_flag_list': [True, True, True, True, True ],
			'var_list': [ b_l, b_r, a, a, e],
			'primal_or_dual': 'dual',
			'conjugateVar': rho,
			}) 
	for k in range(k0+2,n-1):
		Constraint(**{
				'label': f"D_{k}", 
				'sign_list':  [+1, +1, -1, -1],
				'map_list': [tr_l_omega, tr_r_omega , C_l1, C_r1 ]  ,
				'adj_flag_list': [True, True, True, True ],
				'var_list': [g_l[k-1], g_r[k-1], g_l[k], g_r[k]],
				'primal_or_dual': 'dual',
				'conjugateVar': states[k],
				}) 

	dual_constraint_n = Constraint(**{
			'label': f"D_{n}", 
			'sign_list':  [+1, +1,],
			'map_list': [tr_l_omega, tr_r_omega  ]  ,
			'adj_flag_list': [True, True ],
			'var_list': [g_l[n-1], g_r[n-1]],
			'primal_or_dual': 'dual',
			'conjugateVar': states[n],
			}) 

	 
	dual_constraint_n.print_constr_list()
	
	e._close_var_lists()
	
	return rho, tr, dual_constraint_n
