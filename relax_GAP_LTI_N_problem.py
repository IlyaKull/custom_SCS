
import maps
from variables import OptVar
from variables import RestrictedKeysDict
from constraints import Constraint
import matrix_aux_functions as mf
import numpy as np
import util
from CGmaps import iso_from_MPS
import logging
logger = logging.getLogger(__name__)



def set_relax_GAP_LTI_N_problem(n,D, cg_impl = 'kron'):

	logger.info('PROBLEM:  RELAX GAP LTI N')
	
	d=3
	Sx = np.array([[0,1.,0], [1.,0,1.], [0,1.,0]]) / np.sqrt(2)
	iSy = np.array([[0,1.,0], [-1.,0,1.], [0,-1,0]]) / np.sqrt(2)
	Sz = np.array([[1.,0,0], [0,0,0], [0,0,-1.]])
	SdotS = np.kron(Sx, Sx) - np.kron(iSy, iSy) + np.kron(Sz, Sz)
	h =   0.5*( SdotS + ( SdotS @ SdotS )/3 + np.identity(d**2)*2/3 )
	eigsh, UU = np.linalg.eigh(h)
	try:
		assert np.allclose(h, h.T), "h not symmetric"
		assert min(eigsh) > -1e-10, "h not >=0"
	except AssertionError:
		logger.critical("problem with h")
		logger.critical(f"eigs h = {eigsh}")
		raise
	
	A0 = np.sqrt(2/3) * np.array([[0, 1], [0, 0]])
	A1 = -np.sqrt(1/3) * np.array([[1, 0], [0, -1]])
	A2 = -np.sqrt(2/3) * np.array([[0, 0], [1, 0]])
	GSmps = np.array([A0, A1, A2]).transpose((1,0,2))

	
	k0 = util.determine_k0(d, chi = D*D)
	logger.info(f"n = {n}, k0 = {k0}")
	H_terms = dict()
	for l in range(1,k0+3):
		H_terms[(l,l+1)] =mf.tensorProd(np.identity(d**(l-1)), h, np.identity(d**(k0+3-2-l+1)))
		
	H2rho = H_terms[(1,2)] @ H_terms[(1,2)] + mf.anticomm(H_terms[(1,2)], H_terms[(2,3)]) 
		
	for l in range(3,k0+3):
		H2rho += 2.0 *  H_terms[(1,2)] @ H_terms[(l,l+1)]
	
	
	H2omega = 2. * mf.tensorProd(h, np.identity(D*D), h)
		
	logger.info(f'k0 = {k0}, base state: rho_{k0+3}, final state omega_{n}')
	
	assert n >= k0+4 , f'n has to be at least {k0+4}: n = {n}'
	 
	
	
	dims_rho = (d,)*(k0+3)
	dims_omega = (d,d,D*D,d,d)

	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)
	
	# states is a dict with >>key = the number of spins<<
	states = RestrictedKeysDict(allowed_keys = list(range(k0+3,n+1)))
	states[k0+3] = rho
	for k in range(k0+4,n+1):
		states[k] = OptVar(f"omega_{k}",'primal', dims = dims_omega, cone = 'PSD', dtype = float)
	
	# compute isometries for CG
	
	V0, L, R = iso_from_MPS(GSmps, k0, n-4)
	
	# cg maps acting on rho
	action_l0 = {'dims_in': dims_rho, 'pattern':(0,) + (1,)*k0 + (0,0), 'pattern_adj':(0,1,0,0), 'dims_out':(d,D*D,d,d)}
	action_r0 = {'dims_in': dims_rho, 'pattern':(0,0) + (1,)*k0 + (0,), 'pattern_adj':(0,0,1,0), 'dims_out':(d,d,D*D,d)}
	krausOps0 = [V0, ]
	C_l0 = maps.CGmap('C_l0', krausOps0, action = action_l0 , implementation = cg_impl)
	C_r0 = maps.CGmap('C_r0', krausOps0, action = action_r0 , implementation = cg_impl)

	# cg maps acting on omegas 
	action_l1 = {'dims_in': dims_omega, 'pattern':(0,1,1,0,0), 'pattern_adj':(0,1,0,0), 'dims_out':(d,D*D,d,d)}
	action_r1 = {'dims_in': dims_omega, 'pattern':(0,0,1,1,0), 'pattern_adj':(0,0,1,0), 'dims_out':(d,d,D*D,d)}
	
	C_l = [[],]*(n+1)
	C_r = [[],]*(n+1)
	for k in range(k0+4,n):
		C_l[k] = maps.CGmap(f'C_l{k}', [L[k-4],], action = action_l1 , implementation = cg_impl)
		C_r[k] = maps.CGmap(f'C_r{k}', [R[k-4],], action = action_r1 , implementation = cg_impl)

	
	
	tr_l_rho = maps.PartTrace(subsystems = [1], state_dims = dims_rho )
	tr_r_rho = maps.PartTrace(subsystems = [k0+3], state_dims = dims_rho )
	
	tr_l_rho_MINUS_tr_r_rho = maps.AddMaps([tr_l_rho, tr_r_rho], [-1, +1])
			
	tr_l_omega = maps.PartTrace(subsystems = [1], state_dims = dims_omega )
	tr_r_omega = maps.PartTrace(subsystems = [5], state_dims = dims_omega )

	tr_with_H = maps.TraceWith(op_name='H_(1,2)', operator=H_terms[(1,2)], dim=rho.matdim)

	# id_rho = maps.Identity( dim = rho.matdim)
	# id_omega = maps.Identity( dim = states[k0+2].matdim)
	# id_1 = maps.Identity( dim = 1)
	
	# dual varsiables
	a = OptVar('alpha', 'dual', dims = (d,)*(k0+2) , dtype = float )
	b_l = OptVar('beta_l', 'dual', dims = (d,D*D,d,d), dtype = float)
	b_r = OptVar('beta_r', 'dual', dims = (d,d,D*D,d), dtype = float)
	g_l = RestrictedKeysDict(allowed_keys = list(range(k0+3,n))) 
	g_r = RestrictedKeysDict(allowed_keys = list(range(k0+3,n)))

	# beta_l = g_l[k0+3] is dual to V*rho*V^+ == pTr_l omega_k0+4
	# g_l[k0+4] is dual to V*omega_k0+4*V^+ == pTr omega_k0+5 and so on
	g_l[k0+3] = b_l
	g_r[k0+3] = b_r

	for k in range(k0+4,n):
		g_l[k] = OptVar(f"gamma_L_{k}", 'dual', dims = (d,D*D,d,d), dtype = float)
		g_r[k] = OptVar(f"gamma_R_{k}", 'dual', dims = (d,d,D*D,d), dtype = float)

	e = OptVar('epsilon', 'dual', dims = (1,), dtype = float)
	
	
	
	# primal constraints
	# for the sign conventions see sign_convention_doc.txt
	# the LHS in the primal constraints is -A.T
	# the RHS (constants) are without a minus sign,
	# i.e., for a constraint tr(rho)==1 we need to put -tr(rho) in the primal constraints and const = 1
	
	Constraint(**{
		'label': 'norm', 
		'sign_list': [-1,],     # -tr(rho*H_12)
		'map_list': [tr_with_H,],
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
		'label': f'left_{k0+3}', 
		'sign_list': [-1, +1],
		'map_list': [C_l0, tr_l_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, states[k0+4]],
		'primal_or_dual': 'primal',
		'conjugateVar': b_l,
		})
	
	
	Constraint(**{
		'label': f'right_{k0+3}', 
		'sign_list': [-1, +1],
		'map_list': [C_r0, tr_r_omega],
		'adj_flag_list': [False, False],
		'var_list': [rho, states[k0+4]],
		'primal_or_dual': 'primal',
		'conjugateVar': b_r,
		})
	
	for k in range(k0+4,n):
	
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
			'map_list': [C_l0, C_r0, tr_l_rho_MINUS_tr_r_rho, tr_with_H ],
			'adj_flag_list': [True, True, True,  True ],
			'var_list': [ b_l, b_r, a, e],
			'primal_or_dual': 'dual',
			'conjugateVar': rho,
			'const': H2rho.ravel(),
			'const_name' : 'H^2_rho'
			}) 
			
	for k in range(k0+4,n):
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
				'const' : H2omega.ravel(),
				'const_name' : 'H^2_omega'
				}) 

	Constraint(**{
			'label': f"D_{n}", 
			'sign_list':  [+1, +1,],
			'map_list': [tr_l_omega, tr_r_omega  ]  ,
			'adj_flag_list': [True, True ],
			'var_list': [g_l[n-1], g_r[n-1]],
			'primal_or_dual': 'dual',
			'conjugateVar': states[n],
			'const' : H2omega.ravel(),
			'const_name' : 'H^2_omega'
			}) 

	if logging.DEBUG >= logging.root.level:
		Constraint.print_constr_list()
	
	OptVar._close_var_lists()
	
	return  
