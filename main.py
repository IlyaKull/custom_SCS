import line_profiler

import numpy as np
from scs_funcs import solve_M_inv, _apply_M, LinOp_id_plus_AT_A, _id_plus_AT_A
import scs_funcs
from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
import maps
import constraints
import matrix_aux_functions

import relax_LTI_N_problem


def main():

	profile = line_profiler.LineProfiler()
	# profile.add_function(matrix_aux_functions.partial_trace_no_inds)
	profile.add_function(_id_plus_AT_A)
	profile.add_function(LinOp_id_plus_AT_A._matvec)
	profile.add_function(scs_funcs.apply_dual_constr)
	profile.add_function(scs_funcs.apply_primal_constr)
	
	profile.enable_by_count()

	rng = np.random.default_rng(seed=17)
	
	rho, tr, dual_constr_n = relax_LTI_N_problem.set_problem(n=20, D=7, d=2, xOtimesI_impl = 'kron', cg_impl = 'kron')
	
	x, y = rho.initilize_vecs( f_init = rng.random)
	# print(x[:10])
	# print(y[:10])

	 
	lin_op_buff_1 = LinOp_id_plus_AT_A(control_flag =1)

	Mxy_x, Mxy_y = _apply_M(x,y)

	solve_M_inv(Mxy_x,Mxy_y,lin_op_buff_1) #  

	print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

	###########

	lin_op_buff_2 = LinOp_id_plus_AT_A(control_flag =2)

	Mxy_x, Mxy_y = _apply_M(x,y)

	solve_M_inv(Mxy_x,Mxy_y,lin_op_buff_2) #  

	print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

	############
	 
	lin_op_simple =  LinearOperator((OptVar.len_dual_vec_x,)*2, matvec = _id_plus_AT_A )

	Mxy_x, Mxy_y =  _apply_M(x,y)

	solve_M_inv(Mxy_x,Mxy_y,lin_op_simple) #  

	print(max(abs(Mxy_x-x)), max(abs(Mxy_y-y)))

 
	tr.print_maps_log()
	profile.print_stats()
	
if __name__ == '__main__':
	main()
	
