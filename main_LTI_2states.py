
import maps
from variables import OptVar
from constraints import Constraint, print_constraint
import matrix_aux_functions as mf
import numpy as np
import scs_funcs


def main():
	
	d = 2
	D = 3
	dims_rho = [d,d,d,d,d]
	dims_omega = [d,D,d]
	
	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = complex)
	omega = OptVar('omega','primal', dims = dims_omega, cone = 'PSD')
	
	action_l = {'dims_in': dims_rho, 'pattern':[1,1,1,1,0], 'pattern_adj':[1,0], 'dims_out':[D,d]}
	action_r = {'dims_in': dims_rho, 'pattern':[0,1,1,1,1], 'pattern_adj':[0,1], 'dims_out':[d,D]}
	
	krausOps = np.array(np.random.rand(D,d**4), dtype = rho.dtype)
	
	C_l = maps.CGmap('C_l', [krausOps,], action = action_l )
	C_r = maps.CGmap('C_r', [krausOps,], action = action_r )
	
	tr_l_rho = maps.PartTrace(subsystems = [1], state_dims = dims_rho)
	tr_r_rho = maps.PartTrace(subsystems = [5], state_dims = dims_rho)
	
		
	tr_l_omega = maps.PartTrace(subsystems = [1], state_dims = dims_omega)
	tr_r_omega = maps.PartTrace(subsystems = [3], state_dims = dims_omega)
	
	tr = maps.Trace(dim = rho.matdim )
	
 	
	id_rho = maps.Identity( dim = rho.matdim)
	id_omega = maps.Identity( dim = omega.matdim)
	id_1 = maps.Identity( dim = 1)
	
	H_map = maps.TraceWith( 'H', operator = np.identity(rho.matdim, dtype=float) ,dim = rho.matdim )
	
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
	
	
	
	# dual constraints
	signs 		= [+1,]
	operators 	= [id_1,]
	operands	= [e,]
	Constraint('D_obj', signs, operators, operands, 'dual', 'OBJ', add_to_constr_list = False)
	
	signs 		= [+1, +1, +1, +1, -1, -1]
	operators 	= [m.mod_map(adjoint = True) for m in [H_map, C_l, C_r, tr_l_rho, tr_r_rho, id_rho ] ]
	operands	= [None, b_l, b_r, a, a, e]
	constraintd1 = Constraint('D1', signs, operators, operands, 'dual', 'PSD', conjugateVar = rho)
	
	signs 		= [-1, -1]
	operators 	= [m.mod_map(adjoint = True) for m in [tr_l_omega, tr_r_omega ]  ]
	operands	= [b_l, b_r]
	constraintd2 = Constraint('D2', signs, operators, operands  , 'dual', 'PSD', conjugateVar = omega)
	
	constraintd1.print_constr_list()
	
	e._close_var_lists()
	
	###################################################################
	
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'*10)
	
	# vp = np.zeros(OptVar.primal_vars[-1].slice.stop, dtype = rho.dtype)
	# print(f'primal var vec shape = {vp.shape}')
	# rho_init = np.random.rand( rho.matdim, rho.matdim)
	# rho_init = rho_init + rho_init.conj().T
	# omega_init = np.random.rand( omega.matdim, omega.matdim)
	# omega_init = omega_init + omega_init.conj().T
	
	# vp[rho.slice] = mf.mat2vec( dim = rho.matdim, mat = rho_init)
	# vp[omega.slice] = mf.mat2vec( dim = omega.matdim, mat = omega_init)
	
	
	
	vd = np.zeros(OptVar.dual_vars[-1].slice.stop, dtype = a.dtype)
	
	print(f'dual var vec shape = {vd.shape}')
	vd[a.slice] = 1.0
	vd[b_l.slice] = 22.0
	vd[b_r.slice] = 333.0
	vd[e.slice] = 4444.0
	
	A = scs_funcs.LinOp_id_plus_AT_A()
	print(A._matvec(vd))
	print('computed A*vd')
	print('+'*100)
	
	B = scs_funcs.LinOpWithBuffer(matvec = scs_funcs.id_plus_AT_A)
	print(B._matvec(vd))
	
	
	
	'''
	##########################################################################################
	
	
	
	
	
	# print(f"vecs are close: {np.allclose(vp[omega.slice], omega_init[np.triu_indices(np.prod(omega.dims))])}")
	
	# rho_init = np.diag(np.arange(np.prod(rho.dims)) - 10.)
	# rho_init = np.diag(np.arange(np.prod(rho.dims))- 10.)
	
	
	rho_out = mf.vec2mat(rho.matdim, vp[rho.slice])
	omega_out = mf.vec2mat(omega.matdim, vp[omega.slice])
 	 	
	# print(f"omega is symmetric {np.allclose(omega_init, omega_init.T)}")
	# print(f"rho init and rho out are close: {np.allclose(rho_init,rho_out)}")
	# print(f"omega init and omega out are close: {np.allclose(omega_init,omega_out)}")
	# print(f"omega init: \n{omega_init}")
	# print(f"omega out: \n{omega_out}")
	# print(f"diff: \n{omega_init-omega_out}")
	
	print(f"rho vec head: \n{vp[rho.slice][0:16]}")
	
	
	
	# print(f"primal vec = \n{vp}")
	
	 
	# print(f"dual vec = \n{vd}")
	
	
	# print(f"sliced: {vd[b_l.slice].shape}\n indexed: {vd[b_l.indices[0]:b_l.indices[1]+1].shape}")
	
	from scs_funcs import project_to_cone
	tau = -1.
	u = [vd,vp,tau]
	
	import copy
	u_copy = u.copy()
	u_deepcopy = copy.deepcopy(u)
	
	Pu = project_to_cone(u)
	
	print(f"u: \n{u[0][0:16]}\n{u[1][0:16]}\n{u[2]}")
	print(f"u_copy: \n{u_copy[0][0:16]}\n{u_copy[1][0:16]}\n{u_copy[2]}")
	print(f"u_deepcopy: \n{u_deepcopy[0][0:16]}\n{u_deepcopy[1][0:16]}\n{u_deepcopy[2]}")
	print(f"Pu: \n{Pu[0][0:16]}\n{Pu[1][0:16]}\n{Pu[2]}")
	
	# print(f"primal vec = \n{vp}")
	# print(f"dual vec = \n{vd}")
	
	# for c in Constraint.primal_constraints:
		# c(vp, vd)
			
	'''
if __name__ == '__main__':
	main()
