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
import LTI_N_problem, relax_LTI_N_problem, one_step_relax_LTI_N_problem

# problem_module = relax_LTI_N_problem 
problem_module = one_step_relax_LTI_N_problem
# problem_module = LTI_N_problem



def main():
	
	profile_lines = False
	if profile_lines:
		profile = line_profiler.LineProfiler()
		profile.add_function(mf.apply_single_kraus_kron)
		profile.add_function(Maps.__call__)
		profile.enable_by_count()
	
	# print(sys.argv[0])
	chi = int(sys.argv[1])
	maxiter =  int(sys.argv[2])
	
	q = 1.6
	xOtimesI_impl = '4D'
	cg_impl = 'kron'
	
	rng = np.random.default_rng(seed=17)
	
	problem_module.set_problem(chi  = chi , d =2, xOtimesI_impl = xOtimesI_impl, cg_impl = cg_impl)
	# problem_module.set_problem(n=6, d=2, xOtimesI_impl = 'kron') # pure LTI
	
	settings = {'cg_maxiter':	1000,
		'scs_scaling_sigma' : 	0.001,
		'scs_scaling_rho' : 	0.01,
		'scs_q' : 				q,
		'adaptive_cg_iters' : False,
	}
	
	try:
		exact_sol = problem_module.exact_sol
	except AttributeError:
		exact_sol = None

	scs_solver = SCS_Solver(settings , exact_sol = exact_sol)
	scs_solver.run_scs(maxiter = maxiter, printout_every = 50)
	
	
	Maps.print_maps_log()
	
	if profile_lines:
		profile.print_stats()
	
	
if __name__ == '__main__':
	main()
	
