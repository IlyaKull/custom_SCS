import line_profiler

import numpy as np
from scs_funcs import SCS_Solver
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
	
	relax_LTI_N_problem.set_problem(n=8, D=3, d=2, xOtimesI_impl = 'kron', cg_impl = 'kron')

	scs_solver = SCS_Solver(settings = {'cg_maxiter':10000})
	
	
	# scs_solver.run_scs(num_iters=300)
	
	x,y = rng.random((scs_solver.len_dual_vec_x,)) ,rng.random((scs_solver.len_primal_vec_y,))
	Mxy_x, Mxy_y = scs_solver._SCS_Solver__apply_M(x,y)
	sbx, sby = scs_solver._SCS_Solver__solve_M_inv_return(Mxy_x,Mxy_y)
	
	print(f" invert M with _return func: resid = {max(abs(np.concatenate([x,y]) - np.concatenate([sbx,sby]) ) )}"  ) 
	
	ZB
	# scs_solver._solve_M_inv( Mxy_x, Mxy_y)
	
	# print(f" invert M with im-place func: resid = {(max(abs(x-Mxy_x)), max(abs(y-Mxy_y)) )}"  ) 
	
	tao = rng.random((1,))
	# compute 1+Q(x,y,tao) directly to check 1+Q func
	one_plusQ_check = np.zeros(scs_solver.len_joint_vec_u)
	one_plusQ_check[scs_solver.x_slice] = Mxy_x +  tao * scs_solver.c
	one_plusQ_check[scs_solver.y_slice] = Mxy_y +  tao * scs_solver.b
	one_plusQ_check[scs_solver.tao_slice] = tao -np.vdot(y,scs_solver.b) -np.vdot(x,scs_solver.c)
	
	u = np.concatenate([x,y,tao])
	one_plus_Qu = scs_solver._SCS_Solver__one_plus_Q(u)
	print(f"test 1+Q resid = {max(abs(one_plus_Qu - one_plusQ_check))}")
	
	
	
	sbu = scs_solver._SCS_Solver__project_to_affine_return(one_plus_Qu)
	
	print(f" invert 1+Q with _return func: resid = {max(abs(u-sbu))}"  ) 
	
	
	
	sbu2 = np.zeros(scs_solver.len_joint_vec_u)
	scs_solver._project_to_affine(one_plus_Qu, out = sbu2 )
	
	print(f" invert 1+Q with in-place func: resid = {max(abs(u-sbu2))}"  ) 
	
	  
if __name__ == '__main__':
	main()
	
