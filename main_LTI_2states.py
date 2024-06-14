
import maps
from variables import OptVar
from constraints import Constraint, print_constraint
import matrix_aux_functions as mf
import numpy as np
import scs_funcs

# @profile

def main():
	
	d = 2
	D = 3
	dims_rho = (d,d,d,d,d)
	dims_omega = (d,D,d)
	
	rho = OptVar('rho','primal', dims = dims_rho , cone = 'PSD', dtype = float)
	omega = OptVar('omega','primal', dims = dims_omega, cone = 'PSD',dtype = float)
	
	action_l = {'dims_in': dims_rho, 'pattern':(1,1,1,1,0), 'pattern_adj':(1,0), 'dims_out':(D,d)}
	action_r = {'dims_in': dims_rho, 'pattern':(0,1,1,1,1), 'pattern_adj':(0,1), 'dims_out':(d,D)}
	
	krausOps = np.array(np.random.rand(D,d**4), dtype = rho.dtype)
	
	C_l = maps.CGmap('C_l', [krausOps,], action = action_l ,check_inputs = True)
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
	a = OptVar('alpha', 'dual', dims = (d,d,d,d),dtype = float )
	b_l = OptVar('beta_l', 'dual', dims = (D,d),dtype = float)
	b_r = OptVar('beta_r', 'dual', dims = (d,D),dtype = float)
	e = OptVar('epsilon', 'dual', dims = (1,),dtype = float)
	
	
	
	
	
 	
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
	
	
	
	x, y = OptVar.initilize_vecs(np.random.rand)
	print(len(x))
	print(len(y))
	
	
	
	# u = np.zeros(len(vd)+len(vp)+1)
	# u[OptVar.x_slice] = x
	# u[OptVar.y_slice] = y
	# u[OptVar.tao_slice] = 4.
	
	
	
	x2 = np.zeros(len(x))
	scs_funcs.apply_primal_constr(y, x2)
	
	
	
	# lin_op = scs_funcs.LinOp_id_plus_AT_A()
	
	# Mxy_x, Mxy_y = scs_funcs._apply_M(x,y)
	# print(len(Mxy_x))
	# print(len(Mxy_y))
	
	# scs_funcs.solve_M_inv(Mxy_x,Mxy_y,lin_op) #  
	
	# print(np.allclose(Mxy_x,x), np.allclose(Mxy_y,y))
	
	# print(Mxy_x[:10])
	# print(x[:10])
	
	
	# print(Mxy_y[:10])
	# print(y[:10])
	
	
if __name__ == '__main__':
	main()
