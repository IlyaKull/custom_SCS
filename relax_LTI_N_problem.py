
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
		

def set_problem(n,D,d, xOtimesI_impl = 'kron'):

	rng = np.random.default_rng(seed=17)
	
	 
	
	k0 = determine_k0(d,D)
	
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
	C_l0 = maps.CGmap('C_l0', krausOps0, action = action_l0 )
	C_r0 = maps.CGmap('C_r0', krausOps0, action = action_r0 )

	# cg maps acting on omegas 
	action_l1 = {'dims_in': dims_omega, 'pattern':(1,1,1,0), 'pattern_adj':(1,1,0), 'dims_out':(D,D,d)}
	action_r1 = {'dims_in': dims_omega, 'pattern':(0,1,1,1), 'pattern_adj':(0,1,1), 'dims_out':(d,D,D)}
	krausOps1 = [rng.random((D**2,D*D*d)),]
	C_l1 = maps.CGmap('C_l1', krausOps1, action = action_l1 )
	C_r1 = maps.CGmap('C_r1', krausOps1, action = action_r1 )

	
	
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
	 
	
	signs 		= [-1,]
	operators 	= [tr,]
	operands	= [rho,]
	Constraint('norm', signs, operators, operands , 'primal', 'EQ', conjugateVar = e)
	
	signs 		= [-1, +1]
	operators 	= [tr_l_rho, tr_r_rho]
	operands	= [rho, rho]
	Constraint('LTI', signs, operators, operands, 'primal', 'EQ', conjugateVar = a)
	
	signs 		= [-1, +1]
	operators 	= [C_l0, tr_l_omega]
	operands	= [rho, states[k0+2]]
	Constraint(f'left_{k0+1}', signs, operators, operands, 'primal', 'EQ', conjugateVar = b_l)
	
	signs 		= [-1, +1]
	operators 	= [C_r0, tr_r_omega]
	operands	= [rho,states[k0+2]]
	Constraint(f'right_{k0+1}', signs, operators, operands, 'primal', 'EQ', conjugateVar = b_r)
	
	for k in range(k0+2,n):
		signs 		= [-1, +1]
		operators 	= [C_l1, tr_l_omega]
		operands	= [states[k], states[k+1]]
		Constraint(f'left_{k}', signs, operators, operands, 'primal', 'EQ', conjugateVar = g_l[k])
		
		signs 		= [-1, +1]
		operators 	= [C_r1, tr_r_omega]
		operands	= [states[k], states[k+1]]
		Constraint(f'right_{k}', signs, operators, operands, 'primal', 'EQ', conjugateVar = g_r[k])
	
	
	
	# dual constraints
	
	signs 		= [ -1, -1, -1, +1, -1]
	operators0 	= [m.mod_map(adjoint = True) for m in [C_l0, C_r0, tr_l_rho, tr_r_rho, id_rho ] ]
	operands	= [ b_l, b_r, a, a, e]
	Constraint(f"D_{k0+1}", signs, operators0, operands, 'dual', 'PSD', conjugateVar = rho)
	
	
	operators1 	= [m.mod_map(adjoint = True) for m in [tr_l_omega, tr_r_omega , C_l1, C_r1 ]  ]
	for k in range(k0+2,n-1):
		signs 		= [+1, +1, -1, -1]
		
		operands	= [g_l[k-1], g_r[k-1], g_l[k], g_r[k]]
		Constraint(f"D_{k}", signs, operators1, operands  , 'dual', 'PSD', conjugateVar = states[k])
	
	
	signs 		= [+1, +1]
	operators2 	= operators1[:2]
	operands	= [g_l[n-1], g_r[n-1]]
	dual_constraint_n = Constraint(f"D_{n}", signs, operators2, operands  , 'dual', 'PSD', conjugateVar = states[n])
	
	dual_constraint_n.print_constr_list()
	
	e._close_var_lists()
	
	return rho, tr, dual_constraint_n
