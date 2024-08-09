import line_profiler
import sys, os
import numpy as np
from scs_funcs import SCS_Solver
import scs_funcs
# from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
from maps import Maps
# import constraints
import matrix_aux_functions as mf
import LTI_N_problem, relax_LTI_N_problem, one_step_relax_LTI_N_problem, GAP_LTI_N_problem

problem_module = relax_LTI_N_problem 
# problem_module = one_step_relax_LTI_N_problem
# problem_module = LTI_N_problem
# problem_module = GAP_LTI_N_problem

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(levelname)-8s: %(name)-20s:: %(message)s', datefmt='%H:%M:%S')
logging.addLevelName(logging.DEBUG, '...DEBUG') 
logging.addLevelName(5,"verbose")
 

def main():
	
	profile_lines = False
	if profile_lines:
		profile = line_profiler.LineProfiler()
		
		profile.add_function(scs_funcs._impl_apply_constr)
		profile.add_function(scs_funcs.SCS_Solver._project_to_cone)
		
		profile.enable_by_count()

	n = int(sys.argv[1])
	D = int(sys.argv[2])
	maxiter =  int(sys.argv[3])
	
	match sys.argv[4].lower():
		case 'true':
			use_multithread = True
			OMP_NUM_THREADS = 1
			os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
			logger.info(f'MULTITHREADING = TURE, OMP_NUM_THREADS = {OMP_NUM_THREADS}')
		case 'false':
			use_multithread = False
			OMP_NUM_THREADS = 4
			os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
			logger.info(f'MULTITHREADING = FALSE, OMP_NUM_THREADS = {OMP_NUM_THREADS}')
		case _:
			raise Exception("argv[3]: multithreading = 'true' or 'false'")
	
	rng = np.random.default_rng(seed=166).random
	mps = rng((D,2,D))
	
	problem_module.set_problem(n,D, mps)
 	
	settings = {'scs_scaling_sigma' : 	0.001, 	# rescales b
				'scs_scaling_rho' : 	0.01, 	# rescales c
				'scs_q' : 				1.5,
				'adaptive_cg_iters' : True,
				'cg_adaptive_tol_resid_scale' : 20,
				'thread_multithread' : use_multithread, 
				'thread_max_workers' : 4,
				# 'test_maps_SA_tol' : 1e-22
	}
	
	try:
		exact_sol = problem_module.exact_sol
	except AttributeError:
		exact_sol = None

	scs_solver = SCS_Solver(settings , exact_sol = exact_sol)
	scs_solver.run_scs(maxiter = maxiter, printout_every = 100)
	
	if logging.root.level <= 10:
		Maps.print_maps_log()
	
	if profile_lines:
		profile.print_stats()
	
	
if __name__ == '__main__':
	main()
	
