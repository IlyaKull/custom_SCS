
import numpy as np

def make():
	d = {
		'cg_atol' : 1e-12,        # stopping criteria of cg:  norm(b - A @ x) <= max(rtol*norm(b), atol)
		'cg_tol_max' : 1e-3, # when adaptive tolerance is True, cg starts from this tolerance and only decreases see  scs_solver._adapt_cg_tol()
		'cg_tol_Minvh_init' : 1e-14, # tolerance for initial (one-time) solve of M^-1*h
		'cg_maxiter' : 100,
		'cg_adaptive_tol' : True, # adapt the tolerance as iterations progress, see  scs_solver._adapt_cg_tol()
		'cg_adaptive_tol_resid_scale' : 2.5, # tolerance is updated as min(resid[...]/resid_scale), see  scs_solver._adapt_cg_tol()
		'fixed_cg_tol' : 1e-11, # used if adaptive is false
		# 'adaptive_cg_iters' : True,
		# 'cg_adaptive_tol_resid_scale' : 20,
		#
		'log_col_width' : 11, 
		'log_time_func_calls' : True,
		#
		'scs_maxiter': 2000,
		'scs_q' : 1.5,
		'scs_scaling_sigma' : 0.001,  
		'scs_scaling_rho' : 1,  
		'scs_adapt_scale_if_ratio' : 20, # adapts sigma and rho if primal/dual resid ratio is > x or < 1/x  
		'scs_scaling_D': None, # not implemented yet
		'scs_scaling_E': None, # not implemented yet
		'scs_prim_resid_tol' : 1e-6,
		'scs_dual_resid_tol' : 1e-6,
		'scs_gap_resid_tol' : 1e-6,
		'scs_compute_resid_every_n_iters' : 100,
		#
		'test_pos_tol' : 1e-10,
		'test_SA_num_rand_vecs' : 100,
		'test_SA_tol' : 1e-9,
		'test_maps_SA_tol' : 1e-9,
		'test_Minv_h_tol' : 1e-12,
		'test_projToAffine_tol' : 1e-9,
		#
		'thread_multithread' : False,
		'thread_max_workers' : 5,
		#
		'util_rng' : np.random.default_rng(seed=17),
		}
	
	return d
