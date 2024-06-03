
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
	
	rho = OptVar('rho','primal', dims = dims_rho )
	omega = OptVar('omega','primal', dims = dims_omega)
	
	action_l = {'dims_in': dims_rho, 'pattern':[1,1,1,1,0], 'dims_out':[D,d]}
	action_r = {'dims_in': dims_rho, 'pattern':[0,1,1,1,1], 'dims_out':[d,D]}
	C_l = maps.CGmap('C_l', [np.random.rand(D,d**4)], action = action_l )
	C_r = maps.CGmap('C_r', [np.random.rand(D,d**4)], action = action_r )
	
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
	primal_obj = Constraint('P_obj', signs, operators, operands, 'primal', 'OBJ')
	
	signs 		= [+1,]
	operators 	= [id_rho,]
	operands	= [rho,]
	Constraint('pos_rho', signs, operators, operands , 'primal', 'PSD')
	
	signs 		= [+1,]
	operators 	= [id_omega,]
	operands	= [omega,]
	Constraint('pos_omega', signs, operators, operands , 'primal', 'PSD')
	
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
	Constraint('D_obj', signs, operators, operands, 'dual', 'OBJ')
	
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
	
	vp = np.zeros(OptVar.primal_vars[-1].indices_imag[-1])
	print(f'primal var vec shape = {vp.shape}')
	try: 
		imag_ind = OptVar.dual_vars[-1].indices_imag[-1]
	except:
		imag_ind =0
		
	vd = np.ones(max(imag_ind, OptVar.dual_vars[-1].indices_real[-1]))
	print(f'dual var vec shape = {vd.shape}')
	
	cl = Constraint.primal_constraints
	print_constraint(cl[5])
	
	cl[5](vp, vd)
	print(vd)
	print(sum(vd))
	
	
if __name__ == '__main__':
	main()
