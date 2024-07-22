# import line_profiler
import sys
import numpy as np
from scs_funcs import SCS_Solver
import scs_funcs
# from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
from maps import Maps
# import constraints
# import matrix_aux_functions
import LTI_N_problem, relax_LTI_N_problem, one_step_relax_LTI_N_problem

# problem_module = relax_LTI_N_problem 
problem_module = one_step_relax_LTI_N_problem
# problem_module = LTI_N_problem



def main():
	
	# profile = line_profiler.LineProfiler()
	
	# profile.add_function(LinOp_id_plus_AT_A._matvec)
	# profile.add_function(scs_funcs.apply_dual_constr)
	# profile.add_function(scs_funcs.apply_primal_constr)
	# profile.enable_by_count()
	
	# print(sys.argv[0])
	chi = int(sys.argv[1])
	q = float(sys.argv[2])
	
	rng = np.random.default_rng(seed=17)
	
	problem_module.set_problem(chi  = chi , d =2, xOtimesI_impl = 'kron', cg_impl = 'kron')
	# problem_module.set_problem(n=6, d=2, xOtimesI_impl = 'kron') # pure LTI
	
	settings = {'cg_maxiter':	1000,
		'scs_scaling_sigma' : 	0.001,
		'scs_scaling_rho' : 	0.01,
		'scs_q' : 				q,
	}
	
	try:
		exact_sol = problem_module.exact_sol
	except AttributeError:
		exact_sol = None

	scs_solver = SCS_Solver(settings , exact_sol = exact_sol)
	scs_solver.run_scs(maxiter = 5000, printout_every = 20)
	
	
	Maps.print_maps_log()
	  
if __name__ == '__main__':
	main()
	
