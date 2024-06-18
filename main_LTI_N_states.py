
import maps
from variables import OptVar
from variables import RestrictedKeysDict
from constraints import Constraint, print_constraint
import matrix_aux_functions as mf
import numpy as np
import scs_funcs
import sys
from scipy.sparse.linalg import LinearOperator

import relax_LTI_N_problem

# @profile

def main():
	
	relax_LTI_N_problem.set_problem(n=10, D=3, d=2)
	
	x, y = rho.initilize_vecs( f_init = rng.random)
	# print(x[:10])
	# print(y[:10])

	 
	lin_op_buff_1 = scs_funcs.LinOp_id_plus_AT_A(control_flag =1)

	Mxy_x, Mxy_y = scs_funcs._apply_M(x,y)

	scs_funcs.solve_M_inv(Mxy_x,Mxy_y,lin_op_buff_1) #  

	print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

	###########

	lin_op_buff_2 = scs_funcs.LinOp_id_plus_AT_A(control_flag =2)

	Mxy_x, Mxy_y = scs_funcs._apply_M(x,y)

	scs_funcs.solve_M_inv(Mxy_x,Mxy_y,lin_op_buff_2) #  

	print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

	############
	 
	lin_op_simple =  LinearOperator((OptVar.len_dual_vec_x,)*2, matvec = scs_funcs._id_plus_AT_A )

	Mxy_x, Mxy_y = scs_funcs._apply_M(x,y)

	scs_funcs.solve_M_inv(Mxy_x,Mxy_y,lin_op_simple) #  

	print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))


def determine_k0(d,D):
	
	try:
		k0 = next( k for k in range(100) if D**2 < d**k)	
	except StopIteration:
		print('Could not determine k0 value < 100' )
		raise
	
	return k0
		
	
	
if __name__ == '__main__':
	main()
