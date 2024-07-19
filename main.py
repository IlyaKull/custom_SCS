import line_profiler

import numpy as np
from scs_funcs import SCS_Solver
import scs_funcs
# from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
from maps import Maps
# import constraints
# import matrix_aux_functions

# import relax_LTI_N_problem
from LTI_N_problem import set_problem

def main():
	
	# profile = line_profiler.LineProfiler()
	
	# profile.add_function(LinOp_id_plus_AT_A._matvec)
	# profile.add_function(scs_funcs.apply_dual_constr)
	# profile.add_function(scs_funcs.apply_primal_constr)
	# profile.enable_by_count()

	rng = np.random.default_rng(seed=17)
	
	# relax_LTI_N_problem.set_problem(n=8, D=3, d=2, xOtimesI_impl = 'kron', cg_impl = 'kron')
	set_problem(n=6, d=2, xOtimesI_impl = 'kron')
	
	settings = {'cg_maxiter':1000,
		'scs_scaling_sigma' : 0.001,
		'scs_scaling_rho' : 0.01,
	}
	
	try:
		from LTI_N_problem import exact_sol
	except ImportError:
		exact_sol = None

	scs_solver = SCS_Solver(settings , exact_sol = exact_sol)
	scs_solver.run_scs(maxiter = 5000, printout_every = 200)
	
	
	Maps.print_maps_log()
	  
if __name__ == '__main__':
	main()
	
