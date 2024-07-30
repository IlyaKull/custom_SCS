import line_profiler
import sys
import numpy as np
from scs_funcs import SCS_Solver
import scs_funcs
# from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
from maps import Maps
# import constraints
import matrix_aux_functions as mf
import LTI_N_problem, relax_LTI_N_problem, one_step_relax_LTI_N_problem, GAP_LTI_N_problem

# problem_module = relax_LTI_N_problem 
problem_module = one_step_relax_LTI_N_problem
# problem_module = LTI_N_problem
# problem_module = GAP_LTI_N_problem


def main():
	
	profile_lines = True
	if profile_lines:
		profile = line_profiler.LineProfiler()
		profile.add_function(mf.xOtimesI_bc_multi_no_inds)
		profile.add_function(mf.partial_trace_no_inds)
		# profile.add_function(mf.xOtimesI_bc_multi_inds)
		profile.enable_by_count()

	chi = int(sys.argv[1])
	maxiter =  int(sys.argv[2])

	problem_module.set_problem(chi = chi, d =2)
 	
	# maxiter =  int(sys.argv[2])
	# n=   int(sys.argv[1])
	# problem_module.set_problem(n)
 	
	settings = {'scs_scaling_sigma' : 	0.1, 	# rescales b
				'scs_scaling_rho' : 	0.01, 	# rescales c
				'scs_q' : 				1.5,
				'adaptive_cg_iters' : True,
				'cg_adaptive_tol_resid_scale' : 20,
	}
	
	try:
		exact_sol = problem_module.exact_sol
	except AttributeError:
		exact_sol = None

	scs_solver = SCS_Solver(settings , exact_sol = exact_sol)
	scs_solver.run_scs(maxiter = maxiter, printout_every = 100)
	
	
	Maps.print_maps_log()
	
	if profile_lines:
		profile.print_stats()
	
	
if __name__ == '__main__':
	main()
	
