
import maps
from variables import OptVar
from variables import RestrictedKeysDict
from constraints import Constraint
import matrix_aux_functions as mf
import numpy as np
from hamiltonians import make_hamiltonian_term
from matrix_aux_functions import anticomm

def set_problem(n, model = {'name':'AKLT'}):
	"""
	********** problem formulation 
	primal:						| conj Var
	min tr(rho*H2)
	st 	rho >= 0 				: 
		tr(rho*H1) == 1			: delta
		tr_L(rho) == tr_R(rho)	: a
	
	dual:									| conjVar
	max delta
	st	H2 - delta*H1 + Ixa - axI >= 0		: rho 
		
	
	
	*********** scs convention: 
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
	b = H2
	y = rho
	A.T(rho) = [ -tr(rho*H1), -tr_L(rho) + tr_R(rho)]
	c =  [1,0]
	
	
	"""
	
	
	print('>>>>>>>>>>>>>>>>>>> PROBLEM:  GAP LTI N <<<<<<<<<<<<<<<<<<<<<<<<<<')
	
	h_term,d  = make_hamiltonian_term(model)
	
	dims_rho = (d,)*(n)
	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)
	
	H1 = np.kron(h_term, np.identity(d**(n-2))) # extend to size of rho
	
	H2 = np.kron(h_term @ h_term , np.identity(d**(n-2))) 
	H2 += np.kron(\
				anticomm(np.kron(h_term,np.identity(d)) , np.kron(np.identity(d),h_term)),\
				np.identity(d**(n-3))\
			) 
	for j in range(4,n+1):
		H2 += 2* mf.tensorProd(h_term, np.identity(d**(j-4)), h_term, np.identity(d**(n-j))	) 
	
	
	tr_l_rho = maps.PartTrace(subsystems = [1], state_dims = dims_rho )
	tr_r_rho = maps.PartTrace(subsystems = [n], state_dims = dims_rho )
	
	tr_l_rho_MINUS_tr_r_rho = maps.AddMaps([tr_l_rho, tr_r_rho], [+1, -1])
	
	tr_with_H1 = maps.TraceWith(op_name = 'H1', dim = rho.matdim, operator = H1)

	id_rho = maps.Identity( dim = rho.matdim)
	

	# dual varsiables
	a = OptVar('alpha', 'dual', dims = (d,)*(n-1) , dtype = float )
	Q = OptVar('Q', 'dual', dims = dims_rho, dtype = float)
		 
	delta = OptVar('delta', 'dual', dims = (1,), dtype = float)
	
	
	
	# primal constraints
	
	Constraint(**{
		'label': 'H1', 
		'sign_list': [-1,],     # -tr(rho*H1)
		'map_list': [tr_with_H1,],
		'adj_flag_list': [False,],
		'var_list': [rho,],
		'primal_or_dual': 'primal',
		'conjugateVar': delta,
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
		
	
	# dual constraints
	
	Constraint(**{
		'label': 'D_rho', 
		'sign_list':  [ -1, -1],
		'map_list': [tr_l_rho_MINUS_tr_r_rho, tr_with_H1 ],
		'adj_flag_list': [True, True],
		'var_list': [ a, delta],
		'primal_or_dual': 'dual',
		'conjugateVar': rho,
		'const': H2.ravel()
	}) 
	  
 	 
	Constraint.print_constr_list()
	
	OptVar._close_var_lists()
	
	return 0
