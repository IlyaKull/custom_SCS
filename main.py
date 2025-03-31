import line_profiler
import sys, os

import logging
logger = logging.getLogger()
logging.addLevelName(logging.DEBUG, '...DEBUG') 
logging.addLevelName(5,"verbose")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(levelname)-8s: %(name)-20s:: %(message)s', datefmt='%H:%M:%S')
# logger.setLevel(logging.DEBUG)

import numpy as np

import scs_funcs
# from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
from maps import Maps
# import constraints
import matrix_aux_functions as mf
import LTI_N_problem, relax_LTI_N_problem, one_step_relax_LTI_N_problem, GAP_LTI_N_problem

import default_settings

# problem_module = LTI_N_problem
problem_module = relax_LTI_N_problem 
# problem_module = GAP_LTI_N_problem




def main():
		
	# define problem specific command-line arguments
	parser = problem_module.define_arguments()
	
	# add general purpose command line arguments
	parser.add_argument("--verbose", help="increase output verbosity",
						action="store_true")
	parser.add_argument("--profileLines", help="line profiler",
						action="store_true")
	parser.add_argument("--multithread", help="number of threads", 
					type=int, default = 1)
	parser.add_argument("--maxiter", help="maximum number of SCS iterations", 
					type=int, default = 2000)
	parser.add_argument("--dispiters", help="display every x iterations", 
					type=int, default = 100)
	parser.add_argument("--scs_scaling_sigma", help="SCS parameter that rescales b", 
					type=float, default = 0.001)
	parser.add_argument("--scs_scaling_rho", help="SCS parameter that rescales c", 
					type=float, default = 1.0)
	parser.add_argument("--scs_adapt_scale_if_ratio", help="adaptive rescalint if primal to dual residual ratio exceeds this amount", 
					type=float, default = 50.0)

	
					
	args = parser.parse_args()
	
	
	
	if args.verbose:
		logger.setLevel(logging.DEBUG)
	else:
		logger.setLevel(logging.INFO)


	if args.profileLines:
		profile = line_profiler.LineProfiler()
		profile.add_function(scs_funcs._impl_apply_constr)
		profile.add_function(scs_funcs.SCS_Solver._project_to_cone)
		profile.enable_by_count()

	
	use_multithread = args.multithread > 1
	
	if use_multithread:
		OMP_NUM_THREADS = args.multithread
	else:
		OMP_NUM_THREADS = 1
	
	os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
	logger.info(f'MULTITHREADING = {use_multithread}, OMP_NUM_THREADS = {OMP_NUM_THREADS}')
	
	
	settings_from_args = {'scs_scaling_sigma' 		: args.scs_scaling_sigma, 
						'scs_scaling_rho' 			: args.scs_scaling_rho, 
						'scs_adapt_scale_if_ratio' 	: args.scs_adapt_scale_if_ratio,  
						'thread_multithread' 		: use_multithread, 
						'thread_max_workers' 		: OMP_NUM_THREADS,
	}
	
	settings = default_settings.make()  
	settings.update(settings_from_args)
		
	solver = problem_module.set_problem_and_make_solver(args, settings)
 	
	solver.run_scs(maxiter = args.maxiter, printout_every = args.dispiters)
	
	if logging.root.level <= 10:
		Maps.print_maps_log()
	
	if args.profileLines:
		profile.print_stats()
	
	
if __name__ == '__main__':
	main()
	
