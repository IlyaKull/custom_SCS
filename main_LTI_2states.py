
import maps
from variables import OptVar
from constraints import Constraint, print_constraint
import matrix_aux_functions as mf
import numpy as np

def main():
	
	d = 2
	D = 3
	dims_rho = [d,d,d,d,d]
	dims_omega = [d,D,d]
	
	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD')
	omega = OptVar('omega','primal', dims = dims_omega, cone = 'PSD')
	
	action_l = {'dims_in': dims_rho, 'pattern':[1,1,1,1,0], 'dims_out':[D,d]}
	action_r = {'dims_in': dims_rho, 'pattern':[0,1,1,1,1], 'dims_out':[d,D]}
	
	krausOps = np.array(np.random.rand(D,d**4), dtype = rho.dtype)
	
	C_l = maps.CGmap('C_l', [krausOps,], action = action_l )
	C_r = maps.CGmap('C_r', [krausOps,], action = action_r )
	
	tr_l_rho = maps.PartTrace(subsystems = [1], state_dims = dims_rho)
	tr_r_rho = maps.PartTrace(subsystems = [5], state_dims = dims_rho)
	
		
	tr_l_omega = maps.PartTrace(subsystems = [1], state_dims = dims_omega)
	tr_r_omega = maps.PartTrace(subsystems = [3], state_dims = dims_omega)
	
	tr = maps.Trace(dim = np.prod(dims_rho) )
	
	one_var = OptVar('1','primal',dims = (1), add_to_var_list = False)
	
	id_rho = maps.Identity( dim = np.prod(dims_rho))
	id_omega = maps.Identity( dim = np.prod(dims_omega))
	id_1 = maps.Identity( dim = 1)
	
	H_map = maps.TraceWith( 'H', operator = np.identity(np.prod(dims_rho), dtype=complex) ,dim = np.prod(dims_rho) )
	
	# dual varsiables
	a = OptVar('alpha', 'dual', dims = (d,d,d,d) )
	b_l = OptVar('beta_l', 'dual', dims = (D,d))
	b_r = OptVar('beta_r', 'dual', dims = (d,D))
	e = OptVar('epsilon', 'dual', dims = (1,))
	
	
	
	# primal constraints
	signs 		= [+1,]
	operators 	= [H_map,]
	operands	= [rho,]
	primal_obj = Constraint('P_obj', signs, operators, operands, 'primal', 'OBJ', add_to_constr_list = False)
	
	signs 		= [+1,]
	operators 	= [id_rho,]
	operands	= [rho,]
	Constraint('pos_rho', signs, operators, operands , 'primal', 'PSD', add_to_constr_list = False)
	
	signs 		= [+1,]
	operators 	= [id_omega,]
	operands	= [omega,]
	Constraint('pos_omega', signs, operators, operands , 'primal', 'PSD', add_to_constr_list = False)
	
	signs 		= [+1,]
	operators 	= [tr,]
	operands	= [rho,]
	Constraint('norm', signs, operators, operands , 'primal', 'EQ', constant = -1, conjugateVar = e)
	
	signs 		= [+1, -1]
	operators 	= [tr_l_rho, tr_r_rho]
	operands	= [rho, rho]
	Constraint('LTI', signs, operators, operands, 'primal', 'EQ', conjugateVar = a)
	
	signs 		= [+1, -1]
	operators 	= [C_l, tr_l_omega]
	operands	= [rho, omega]
	Constraint('left', signs, operators, operands, 'primal', 'EQ', conjugateVar = b_l)
	
	signs 		= [+1, -1]
	operators 	= [C_r, tr_r_omega]
	operands	= [rho,omega]
	Constraint('right', signs, operators, operands, 'primal', 'EQ', conjugateVar = b_r)
	
	e.print_var_list()
	
	# dual constraints
	signs 		= [+1,]
	operators 	= [id_1,]
	operands	= [e,]
	Constraint('D_obj', signs, operators, operands, 'dual', 'OBJ', add_to_constr_list = False)
	
	signs 		= [+1, +1, +1, +1, -1, -1]
	operators 	= [m.mod_map(adjoint = True) for m in [H_map, C_l, C_r, tr_l_rho, tr_r_rho, id_rho ] ]
	operands	= [one_var, b_l, b_r, a, a, e]
	constraintd1 = Constraint('D1', signs, operators, operands, 'dual', 'PSD', conjugateVar = rho)
	
	signs 		= [-1, -1]
	operators 	= [m.mod_map(adjoint = True) for m in [tr_l_omega, tr_r_omega ]  ]
	operands	= [b_l, b_r]
	constraintd2 = Constraint('D2', signs, operators, operands  , 'dual', 'PSD', conjugateVar = omega)
	
	constraintd1.print_constr_list()
	
	
	
	##########################################################################################
	
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'*10)
	
	vp = np.zeros(OptVar.primal_vars[-1].slice.stop, dtype = rho.dtype)
	print(f'primal var vec shape = {vp.shape}')
	# rho_rand = np.random.rand( np.prod(rho.dims), np.prod(rho.dims))
	# rho_rand = rho_rand + rho_rand.conj().T
	# omega_rand = np.random.rand( np.prod(omega.dims), np.prod(omega.dims))
	# omega_rand = omega_rand + omega_rand.conj().T
		
	# vp[rho.indices[0]:rho.indices[1]+1] = mf.mat2vec( dim = np.prod(rho.dims), mat = rho_rand)
	# vp[omega.indices[0]:omega.indices[1]+1] = mf.mat2vec( dim = np.prod(omega.dims), mat = omega_rand)
	# print(f"primal vec = \n{vp}")
	
	 
	
	vd = np.zeros(OptVar.dual_vars[-1].slice.stop, dtype = a.dtype)
	
	print(f'dual var vec shape = {vd.shape}')
	vd[a.slice] = 1.0
	vd[b_l.slice] = 22.0
	vd[b_r.slice] = 333.0
	vd[e.slice] = 4444.0
	print(f"dual vec = \n{vd}")
	
	
	# print(f"sliced: {vd[b_l.slice].shape}\n indexed: {vd[b_l.indices[0]:b_l.indices[1]+1].shape}")
	
	# from scs_funcs import proj_to_cone
	# tau = -1.
	# u = [vd,vp,tau]
		
	# project_to_cone(u)
	
	# print(f"primal vec = \n{vp}")
	# print(f"dual vec = \n{vd}")
	
	# for c in Constraint.primal_constraints:
		# c(vp, vd)
			
	
if __name__ == '__main__':
	main()
