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
	
	relax_LTI_N_problem.set_problem(n=20, D=3, d=2, xOtimesI_impl = 'kron', cg_impl = 'kron')

	scs_solver = SCS_Solver(settings = {})

	
	u = scs_solver.u
	one_plus_Qu = scs_solver._SCS_Solver__one_plus_Q(u)
	
	sbu = scs_solver._SCS_Solver__project_to_affine_return(one_plus_Qu)
	
	print(max(abs(u-sbu)))
	
	scs_solver._project_to_affine(one_plus_Qu)
	
	print(max(abs(u-one_plus_Qu)))
	
	  
if __name__ == '__main__':
	main()
	
