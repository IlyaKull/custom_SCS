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


'''
matlab code:
--------------------------
rho2 = sdpvar(d^2);
rho2x2 = sdpvar(d^4);

rho4x2 = sdpvar(d^(4*2));
% 
%   1,5
%   2,6 
%   3,7
%   4,8
% 
omega4x4 = sdpvar(D(2)^(2*4));
%   1',2',3',4' 
%   5',6',7',8'
% 
% (1,2)->1'     (5,6)->2' 
% (3,4)->5'     (7,8)->6'

omega8x4 =  sdpvar(D(3)^(2*4));

%   1'',5''
%   2'',6'' 
%   3'',7''
%   4'',8''

% (1',2')->1''     (5',6')->2'' 
% (3',4')->5''     (7',8')->6''
t=tic;
fprintf('Setting constraints...')

constraints = [ rho4x2 >= 0,...
                omega4x4 >= 0,...
                omega8x4 >= 0,...
                %
                trace(rho2) == 1,...
                %
                pTr(rho2x2,[3,4],[d,d,d,d]) == rho2,...
                pTr(rho2x2,[1,2],[d,d,d,d]) == rho2,...
                pTr(rho2x2,[1,3],[d,d,d,d]) == rho2,...
                pTr(rho2x2,[2,4],[d,d,d,d]) == rho2,...
                %
                pTr(rho4x2,[1,5,4,8],d*ones(1,8)) == rho2x2,...
                %
                pTr(rho4x2,[1,5], d*ones(1,8)) == pTr(rho4x2,[4,8], d*ones(1,8)),...    
                pTr(omega4x4,[1,5], D(2)*ones(1,8)) == pTr(omega4x4,[4,8], D(2)*ones(1,8)),...
                pTr(omega8x4,[1,5], D(3)*ones(1,8)) == pTr(omega8x4,[4,8], D(3)*ones(1,8)),...
                %
                apply4CG_iter(krausOps{1},rho4x2)   == syspermute(pTr(omega4x4,[3 4 7 8], D(2)*ones(1,8)),[1 3 2 4],D(2)*ones(1,4)),... 
                apply4CG_iter(krausOps{2},omega4x4) == syspermute(pTr(omega8x4,[3 4 7 8], D(3)*ones(1,8)),[1 3 2 4],D(3)*ones(1,4))... 
              ];
'''




def define_arguments():
	'''
	define the command line arguments required to define this problem
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("num_levels", help="number of levels: 1 is the base level with an 8-body state, 2 has a 16-body state and so on", type=int)
	parser.add_argument('--cg_dims', type=list, action='store', dest='list',
                      help='<Required> list of cg dimensions, length of the list should be num_levels-1',
                      required=True)
	# ~ parser.add_argument("D", help="bond dim of MPS used for compression", type=int)
	# ~ parser.add_argument("--mps_filename", help="file where compression mps is saved", type=str, default = '')
	# ~ parser.add_argument("--coarse_grain_method", help="which coarse graining maps to construct from given MPS", 
					# type=str, default = "plain", choices=["plain", "isometries"])

	return parser
	


def set_problem_and_make_solver(args, settings):

	logger.info('PROBLEM:  TWO DIM TREE COARSE GRAINING	')
	
	num_levels = args.num_levels
	cg_dims = args.cg_dims
	
	assert len(cg_dims) == num_levels-1
	
	
	d=2
		# todo: load isometries/unitaries from file
	# if len(args.mps_filename)>0:
		# mps = read_mps_from_file(args.mps_filename)
		# try:
			# assert mps.shape == (D,d,D)
		# except AssertionError:
			# logger.critical(f'loaded mps dimensions {mps.shape} are incompatible with problem dimensions {(D,d,D)}')
			# raise
	# else:
		# logger.critical("no MPS file specified!!! proceeding with random MPS")
		# rng = settings['util_rng']
		# mps = rng.random((D,d,D,)) 
		
	
	# heisenberg model with sub-lattice rotation -H_xxz(1,1,-1)
	h_term = np.array(  [[0.25, 0, 0, 0],
					[0, -0.25, -0.5, 0],
					[0, -0.5, -0.25, 0],
					[0, 0, 0, 0.25]]
				)
	
	
	variational_sol = -0.669437/2  # mcmc from  doi 10.1103/PhysRevB.56.11678
	
	
	H = ??? #np.kron(h_term, np.identity(d**(k0+1-2))) # extend to size of rho
	
	
	
	'''
	Define the states, i.e. the primal variables:
	
	first level has rho_8 = rho_4x2 as the biggest state
	rho_8 := rho4x2
	 
	   1,5
	   2,6 
	   3,7
	   4,8
	
	The second level is a coarse graining of a state on a 4x4 patch:
	rho_16 := rho_4x4 		omega16 := omega4x4
	 1 5 9  13       	1',2',3',4' 
	 2 6 10 14  --CG--> 	
	 3 7 11 15			5',6',7',8'
	 4 8 12 16
	
	where the same coarse graining map is applied to each pair:
	 (1,2)->1'     (5,6)->2' 
	 (3,4)->5'     (7,8)->6'
	
	The third level incorporates a twice coarse grainde state on a 8x4 patch
	
	omega_32 := omega8x4

	%   1'',5''
	%   2'',6'' 
	%   3'',7''
	%   4'',8''

	% (1',2')->1''     (5',6')->2'' 
	% (3',4')->5''     (7',8')->6''
	'''
		
	dims_rho = (d,)*8
	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)
	
	
	# dictionaries with >>key = the number of spins<< 
	num_spins_per_level = [2**l for l in range(3, 2+num_levels+1)] # one level: [8], two levels: [8,16], three: [8,16,32] and so on
	
	dims_cg_states = RestrictedKeysDict(allowed_keys = num_spins_per_level) 
	states = RestrictedKeysDict(allowed_keys = num_spins_per_level) 
	
	dims_cg_states[8] = dims_rho
	states[8] = rho
	for i,k in enumerate(num_spins_per_level[1:]):
		dims_cg_states[k] = (cg_dims[i],) *8
		states[k] = OptVar(f"omega_{k}", 'primal', dims = dims_cg_states[k], cone = 'PSD', dtype = float)

	'''
	************************************
	note about 
	THE SITE INDEXING CONVETION:
	
	rho_8 is obtained from rho_16 in 3 different ways:
	trace_{9,...16}, trace_{1,...,4, 13,...,16} and trace_{1,...8}.
	Equivalently, we can demand that rho_16 is LTI with respect to horizontal translations
	
	after coarse graining the relations rho_8 = tr_{~}(rho_16) we obtain the relation
	trace_{1',5',2',6'} omega_16 = cg_{1,2} x cg_{3,4} x cg_{5,6} x cg_{7,8} rho_8
								   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    that is: with this indexing convention we only need to apply cg maps on consecutive pairs of spins.
    The functions for implementing cg maps support such operations.
    The partial trace operations require permutations, which are supported in the implementation.
	
	************************************
	'''
		
	
	 
	
	# cg maps
	
	cg_impl = 'contract' #'kron' implementation inapplicable in case of multiple maps acting simultaneously 
	
	cg_action_rho = {	'dims_in'		: 	dims_rho,
						'pattern'		: 	(1,1, 2,2, 3,3, 4,4),
						'pattern_adj'	:	(1,2,3,4),
						'dims_out'		:	(cg_dims[0],)*4
					}
	
	cg_actions = RestrictedKeysDict(allowed_keys = num_spins_per_level) 
	kraus_ops = RestrictedKeysDict(allowed_keys = num_spins_per_level) 
	cg_actions[8] = cg_action_rho
	
	for i,k in enumerate(num_spins_per_level[1:]):
		cg_actions[k] = {	'dims_in'		: 	dims_cg_states[k],
							'pattern'		: 	(1,1, 2,2, 3,3, 4,4),
							'pattern_adj'	:	(1,2,3,4),
							'dims_out'		:	(cg_dims[i],)*4
						}

	
	cg_maps = RestrictedKeysDict(allowed_keys = num_spins_per_level) 
	for i,k in enumerate(num_spins_per_level[:-1]): # cg maps act on all states but the last
		kraus_ops[k] = None # TODO  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		cg_maps[k] = maps.CGmap(f'C_{k}', kraus_ops[k], action = cg_actions[k] , implementation = cg_impl)
	
	
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
	
	solver = SCS_Solver(settings, variational_sol = variational_sol)
	
	return solver
	


def read_mps_from_file(mps_filename):
	try:
		with open(mps_filename, 'rb') as f:
			mps = np.load(f)
	except FileNotFoundError:
		logger.critical(f"specified file '{mps_filename}' not found")
		raise
		
	return mps
	
