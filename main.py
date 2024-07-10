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
	
	print(f" invert M with _return func: resid = {(max(abs(x-sbx)), max(abs(y-sby)) )}"  ) 
	
	scs_solver._solve_M_inv( Mxy_x, Mxy_y)
	
	print(f" invert M with im-place func: resid = {(max(abs(x-Mxy_x)), max(abs(y-Mxy_y)) )}"  ) 
	
	
	
	
	u1 = scs_solver.u
	u2 = rng.random((scs_solver.len_joint_vec_u,))
	
	u = u1 + 0.1 * u2
	one_plus_Qu = scs_solver._SCS_Solver__one_plus_Q(u)
	
	sbu = scs_solver._SCS_Solver__project_to_affine_return(one_plus_Qu)
	
	print(f" invert 1+Q with _return func: resid = {max(abs(u-sbu))}"  ) 
	
	sbu2 = np.zeros(scs_solver.len_joint_vec_u)
	scs_solver._project_to_affine(one_plus_Qu, out = sbu2 )
	
	print(f" invert 1+Q with in-place func: resid = {max(abs(u-sbu2))}"  ) 
	
	  
if __name__ == '__main__':
	main()
	
