

import numpy as np
from scs_funcs import solve_M_inv, _apply_M, LinOp_id_plus_AT_A, _id_plus_AT_A
from  variables import OptVar
from scipy.sparse.linalg import LinearOperator

import relax_LTI_N_problem

# @profile

def main():
	
	rng = np.random.default_rng(seed=17)
	
	rho, tr, dual_constr_n = relax_LTI_N_problem.set_problem(n=20, D=4, d=2, xOtimesI_impl = 'kron')
	
	x, y = rho.initilize_vecs( f_init = rng.random)
	# print(x[:10])
	# print(y[:10])

	 
	lin_op_buff_1 = LinOp_id_plus_AT_A(control_flag =1)

	Mxy_x, Mxy_y = _apply_M(x,y)

	solve_M_inv(Mxy_x,Mxy_y,lin_op_buff_1) #  

	print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

	###########

	# lin_op_buff_2 = LinOp_id_plus_AT_A(control_flag =2)

	# Mxy_x, Mxy_y = _apply_M(x,y)

	# solve_M_inv(Mxy_x,Mxy_y,lin_op_buff_2) #  

	# print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

	# ############
	 
	# lin_op_simple =  LinearOperator((OptVar.len_dual_vec_x,)*2, matvec = _id_plus_AT_A )

	# Mxy_x, Mxy_y =  _apply_M(x,y)

	# solve_M_inv(Mxy_x,Mxy_y,lin_op_simple) #  

	# print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

 
	tr.print_maps_log()
	
if __name__ == '__main__':
	main()
