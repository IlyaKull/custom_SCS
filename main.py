import line_profiler

import numpy as np
from scs_funcs import solve_M_inv, _apply_M, LinOp_id_plus_AT_A, _id_plus_AT_A, _one_plus_Q, project_to_affine
import scs_funcs
from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
import maps
import constraints
import matrix_aux_functions

import relax_LTI_N_problem


def main():
	
	# profile = line_profiler.LineProfiler()
	
	# profile.add_function(LinOp_id_plus_AT_A._matvec)
	# profile.add_function(scs_funcs.apply_dual_constr)
	# profile.add_function(scs_funcs.apply_primal_constr)
	# profile.enable_by_count()

	rng = np.random.default_rng(seed=17)
	
	rho, tr, dual_constr_n = relax_LTI_N_problem.set_problem(n=20, D=3, d=2, xOtimesI_impl = 'kron', cg_impl = 'kron')
	
	x, y = rho.initilize_vecs( f_init = rng.random)
	# print(x[:10])
	# print(y[:10])

	# test Minv
	lin_op_buff_1 = LinOp_id_plus_AT_A()

	Mxy_x, Mxy_y = _apply_M(x,y)
	solx, soly = scs_funcs._solve_M_inv_return(Mxy_x,Mxy_y,lin_op_buff_1)
	print(f"Minv resid 1 = {max(abs(solx-x)), max(abs(soly-y))}" )
	
	solve_M_inv(Mxy_x,Mxy_y,lin_op_buff_1) #  
		
	print(f"Minv resid 1 = {max(abs(Mxy_x-x)), max(abs(Mxy_y-y))}" )
	
	
	
	# test 1+Q inv
	u = np.zeros(OptVar.tao_slice.stop);
	u[OptVar.x_slice], u[OptVar.y_slice] = rho.initilize_vecs( f_init = rng.random)
	u[OptVar.tao_slice] = rng.random(1)
	
	c, b = rho.initilize_vecs( f_init = rng.random)
	IplusQ_u = _one_plus_Q(u,c,b)
	
	h = np.zeros(OptVar.y_slice.stop)
	h[OptVar.x_slice] = c
	h[OptVar.y_slice] = b
	 
	Minvh = np.zeros(OptVar.y_slice.stop)	
	xx,yy = scs_funcs._solve_M_inv_return(c,b,lin_op_buff_1)
	Minvh[OptVar.x_slice] = xx
	Minvh[OptVar.y_slice] = yy
	
	MMinvh_x, MMinvh_y = _apply_M(Minvh[OptVar.x_slice], Minvh[OptVar.y_slice])
	print(f"M*Minv*h -h resid = {max( [max(abs(MMinvh_x-c)), max(abs(MMinvh_y-b))] )}")
	
	# print(Minvh)
	
	
	hMinvh_plus_one_inv = 	1.0/(1.0 + np.vdot(h,Minvh))
	
	sbu = scs_funcs._project_to_affine_return(IplusQ_u, lin_op_buff_1,  c,  b, hMinvh_plus_one_inv, Minvh)
	print(f'inverted 1+Q max resid: {max(abs(u-sbu))}')
	
	project_to_affine(IplusQ_u, lin_op_buff_1,  c,  b, hMinvh_plus_one_inv, Minvh)
	print(f'inverted 1+Q max resid: {max(abs(u-IplusQ_u))}')
	
	
 
	# tr.print_maps_log()
	# profile.print_stats()
	
if __name__ == '__main__':
	main()
	
