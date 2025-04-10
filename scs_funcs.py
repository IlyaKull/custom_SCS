import numpy as np
from variables import OptVar
from maps import Maps
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg as scipy_cg
# from scipy.sparse.linalg import eigsh as scipy_eigsh
from constraints import Constraint
import time
import util
import concurrent.futures
import default_settings

import logging
logger = logging.getLogger(__name__)


class SCS_Solver:
	'''
	after defining all the optimization variables (OptVar class),
	and all the constraints on them (expressions involving Maps acting on OptVars)
	the problem has been specified.
	
	At this point the SCS_Solver class can be instantiated. 
	It contains all the methods and objects needed to perform the scs algorithm.
	https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 
	
	
	When initialized this class, it first closes the variable lists in the OptVar calss and
	 retrieves all the calss variables from OptVar and Constraint classes, 
	 such that all the data neede for the algorithm is encapsulated in this object. 
	
	variable names based on 
	https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 3.2.3 and 3.3
	
	'''
	def __init__(self, settings = dict(), exact_sol = None):
		
		logger.info(f'initializing scs solver')
		if not OptVar.lists_closed:  # closing the lists fixes the sizes of the primal and dual vectors
			OptVar._close_var_lists()
			
		
		self.settings = default_settings.make()  
		self.settings.update(settings)
		 
		
		self.exact_sol = exact_sol
		logger.debug(f"importing constraints lists from 'Constraint' calss (constraints.py)")		
		self.primal_constraints = Constraint.primal_constraints
		self.dual_constraints = Constraint.dual_constraints
		
		logger.debug(f"importing variable lists from OprVar calss (variables.py)")
		self.len_primal_vec_y = OptVar.len_primal_vec_y
		self.len_dual_vec_x = OptVar.len_dual_vec_x
		
		self.x_slice = OptVar.x_slice
		self.y_slice = OptVar.y_slice
		self.tao_slice = OptVar.tao_slice
		
		self.len_joint_vec_u = self.len_dual_vec_x + self.len_primal_vec_y + 1
		
		self.dtype = OptVar.dtype
		
		self.primal_vars = OptVar.primal_vars
		self.dual_vars = OptVar.dual_vars
		self.lists_closed = OptVar.lists_closed
		
		# initialize iteration vectors:
		self.u = self._initilize_vec()
		self.v = self._initilize_vec()
		self.u_tilde = self._initilize_vec()
		
		# the q parameter in the iteration
		self.q = self.settings['scs_q']
		
		
		logger.debug("initializing linear operator") 
		self.lin_op = LinOp_id_plus_AT_A(self.len_dual_vec_x,\
			self.len_primal_vec_y,\
			self.dtype,\
			self.primal_constraints,\
			self.dual_constraints,\
			self.settings
		)
		
		self._test_constraints_validity()
		
		self._read_b_and_c()
		# the vector h is defined as h = [c,b]
	
				
		# M^(-1)h is needed in every iteration and is therefore computed
		# at initialization 
		self.cg_iter_counter = 0
		
		Minv_h = np.zeros(self.len_dual_vec_x + self.len_primal_vec_y)
		Minv_h[self.x_slice], Minv_h[self.y_slice] = self.__solve_M_inv_return(self.c, self.b, tol = self.settings['cg_tol_Minvh_init'])
 
		# the following is also needed in every iteration	
		self.hMinvh_plus_one_inv = 	1.0/(1.0 + np.vdot(self.h, Minv_h))
		self.Minv_h = Minv_h
		
		self._test_Minv_h(tol = self.settings['test_Minv_h_tol'])
		
		self.cg_tol = self.settings['cg_tol_Minvh_init']
		
		self.t_affine = 0.0
		self._test_projToAffine()
		
		#reset timers and counters
		self.t_affine = 0.0
		self.t_cone = 0.0
		self.iter_counter_for_cg_avg = 0
		self.cg_iter_counter = 0
		self.cg_avrg_iter_count = 0
		
		
		# progress log tracks the following 
		self.resid_prim = None
		self.resid_dual = None
		self.resid_gap = None
		self.primal_objective = None
		self.dual_objective = None
		
		logger.info(f'solver initialized')
		
			
		
	def _test_constraints_validity(self):
		'''
		When the constraints are correctly defined (dual is indeed the dual of primal) then 
		1. applying the dual constraints should be the adjoint operation of applying the primal.
		2. linear operator 1+A^T*A should always be >= 1.
		'''
		
		logger.info("TESTING: constraints validity")
		logger.info("TESTING: maps table:")
		# first complete the check of maps_table to make sure that the dual constraints that were input are indeed the dual of the primal ones
		Constraint._check_maps_table()
		
		# check each map individualy
		logger.info("TESTING: check all maps implementations M.apply_adj() is adj of M.apply():")
		try:
			violations = Maps._test_self_adjoint(rng = self.settings['util_rng'],\
				n_samples = 20,\
				tol = self.settings['test_maps_SA_tol'])
		except AssertionError:
			raise # assertion in above _test_self_adjoint func
		else:
			logger.info("All maps checked")
			
		
		logger.info("TESTING: check that dual constraints are the adjiont of primal constraints (vdot(y, Ax) == vdot(A.Ty, x))")
		rng = self.settings['util_rng']
		 
		sa_tests = np.zeros(self.settings['test_SA_num_rand_vecs'])
		for j in range(self.settings['test_SA_num_rand_vecs']):
			x,y = rng.random((self.len_dual_vec_x,)) ,rng.random((self.len_primal_vec_y,), dtype = self.dtype)
			x = x/np.linalg.norm(x)
			y = y/np.linalg.norm(y)
			Ax = apply_dual_constr(self, x)
			ATy = apply_primal_constr(self, y)
			sa_tests[j] = np.vdot(y, Ax) - np.vdot(ATy, x)
		
		max_violation_SA = max(abs(sa_tests))
		try:
			assert max_violation_SA < self.settings['test_SA_tol'],	f"primal and dual constraints are not adjoints of each other."
		except AssertionError as SAerr:
			logger.critical(f"Max violation of vdot(y, Ax) - vdot(A.Ty, x) is {max_violation_SA : 0.3g}, tolerance {self.settings['test_SA_tol'] :0.3g}")
			raise SAerr
		else:
			logger.info(f"Self adjoint test passed. max violation is {max_violation_SA : 0.3g}, tolerance {self.settings['test_SA_tol'] :0.3g}")
		
		
		
		
	def _test_Minv_h(self, tol = None):
		if tol is None: 
			tol = self.settings['test_Minv_h_tol']
			
		logger.info("TESTING: Minv_h (M * Minvh == h):")
		sb_hx, sb_hy = self.__apply_M(self.Minv_h[self.x_slice], self.Minv_h[self.y_slice])
		resid = max(abs(np.concatenate([sb_hx, sb_hy]) - self.h))
		try:
			assert resid < tol , 'Minv_h test failed'
		except AssertionError:
			logger.critical(f"max|M*Minv_h - h resid| = {resid} > tol = {tol}")
			raise
		else:
			logger.info("Minv_h passed")
		
		
	def _test_projToAffine(self, tol = None):
		if tol is None: 
			tol = self.settings['test_projToAffine_tol']
		rng = self.settings['util_rng']
			
		
		tao = rng.random((1,))
		x,y = rng.random((self.len_dual_vec_x,)) ,rng.random((self.len_primal_vec_y,), dtype = self.dtype)
		u = np.concatenate([x,y,tao])
		u = u/np.linalg.norm(u)
		
		one_plus_Qu = self.__one_plus_Q(u)
		
		if False: #tests from early debugging 
			# test that project to affine method inverts 1+Q as it should
			logger.info("TESTING: _project_to_affine() = (1+Q)^-1 :")
			logger.debug("test _return method")
			sbu1 = self.__project_to_affine_return(one_plus_Qu)
			resid1 = max(abs(u-sbu1))
			try:
				assert resid1 < tol , "test _project_to_affine failed"
			except AssertionError:
				logger.critical(f"while testing invert 1+Q with _return func: resid = {resid1} > tol = {tol}")
				raise
			else:
				logger.debug(f"test _project_to_affine_return passed")
				
		# test that project to affine method inverts 1+Q as it should
		logger.debug("test _project_to_affine")	
		sbu2 = np.zeros(self.len_joint_vec_u)
		self._project_to_affine(one_plus_Qu, out = sbu2 )
		resid2 = max(abs(u-sbu2))
		try:
			assert resid2 < tol, "test _project_to_affine failed"
		except AssertionError:
			logger.critical(f"while testing invert 1+Q with in-place func: resid = {resid2} > tol = {tol}")
			raise
		else:
			logger.info(f"test _project_to_affine passed")
		
	def _read_b_and_c(self):	
		'''
		the vector h is defined as h = [c,b]
		where b is the vector of constants in the constraint Ax+s=b, s>= 0
		and c is the objective function: min(c@x)
		see sign_conventions.txt
		'''
		
		self.h = np.zeros(self.len_dual_vec_x + self.len_primal_vec_y)
		self.c = self.h[self.x_slice] 
		self.b = self.h[self.y_slice]
		for cstr in self.primal_constraints:
			self.c[cstr.conjugateVar.slice] = cstr.const
		for cstr in self.dual_constraints:
			self.b[cstr.conjugateVar.slice] = cstr.const
		
		# needed for termination criteria (before rescaling)
		self.b_unscld_norm = np.linalg.norm(self.b)
		self.c_unscld_norm = np.linalg.norm(self.c)
		
		# RESCALE
		self.sigma = self.settings['scs_scaling_sigma']
		self.rho = self.settings['scs_scaling_rho']
		self.b *= self.sigma
		self.c *= self.rho
		 
		logger.debug(f"RESCALING: b --> sigma * b ; (sigma = {self.sigma :0.3g})")
		logger.debug(f"RESCALING: c --> rho * c ;   (rho = {self.rho :0.3g})")
		
		
		
	
	
	def run_scs(self, maxiter = None, printout_every = None):
		
		if maxiter is None:
			maxiter = self.settings['scs_maxiter'] 
		else: 
			self.settings['scs_maxiter']  = maxiter
		
		if printout_every is None:
			printout_every = self.settings['scs_compute_resid_every_n_iters'] 
		else:
			self.settings['scs_compute_resid_every_n_iters'] = printout_every
		
		self.cg_tol = self.settings['cg_tol_max']
		self.iter = 0
		termination_criteria_satisfied = False
		self.t_start = time.perf_counter()
		
		self._print_log(print_head = True, print_line = False)
		
		#iteration
		while (self.iter < maxiter) and (not termination_criteria_satisfied):
			self._iterate_scs()
			self.iter += 1
			
			if self.iter % printout_every == 0 or self.iter == maxiter:
				termination_criteria_satisfied = self._check_termination_criteria()
				self._print_log(converged = termination_criteria_satisfied)
				self._adapt_cg_tol()
				self._adapt_scaling()
		
		if not termination_criteria_satisfied:
			self._print_log(maxed_out_iters = True)
		
	def _adapt_cg_tol(self):
		if self.settings['cg_adaptive_tol']:
			self.settings['cg_tol_max'] = self.cg_tol
			scl = self.settings['cg_adaptive_tol_resid_scale']
			self.cg_tol = min([self.resid_prim/scl, self.resid_dual/scl, self.resid_gap/scl, self.settings['cg_tol_max'] ])	
		else:
			self.cg_tol = self.settings['fixed_cg_tol']
		
			
	
	def _print_log(self, print_head = False, print_line = True, converged = False, maxed_out_iters = False):
				
		col_width = self.settings['log_col_width']
		 
		res_format_str = '0.5g'
		obj_format_str = '0.5g'
		
		log_columns = {"Iter" : 	{'val' : self.iter, 			'format_str' : 'g'},
			"Prim res" : 			{'val' : self.resid_prim, 		'format_str' : res_format_str},
			"Dual res" : 			{'val' : self.resid_dual, 		'format_str' : res_format_str},
			"Gap res" : 			{'val' : self.resid_gap, 		'format_str' : res_format_str},
			"Prim obj" :			{'val' : self.primal_objective, 'format_str' : obj_format_str},
			"Dual obj" : 			{'val' : self.dual_objective, 	'format_str' : obj_format_str},
			"Tot time":				{'val' : time.perf_counter() - self.t_start, 'format_str' : '0.3g'},
			"Aff time":				{'val' : self.t_affine, 'format_str' : '0.3g'},
			"Cone time":			{'val' : self.t_cone, 'format_str' : '0.3g'},
			"avg cg iter":			{'val' : self.cg_avrg_iter_count, 'format_str' : 'g'},
		}
		
		if print_head:
			print("="*100)
			print("STARTING SCS".center(100))
			print("="*100)

			for k in log_columns:
				print(k.ljust(col_width), end = " | " )
			print('')
			
		if print_line:
			for dikt in log_columns.values():
				print(format(dikt['val'], dikt['format_str']).ljust(col_width), end = " | " )
			print('')
		
		if converged or maxed_out_iters:
			if converged:
				print(f'==================> converged')
					
			if maxed_out_iters:
				print(f"==================> not converged, reached maxiter ({self.settings['scs_maxiter'] })")
			
			if not self.exact_sol is None:
				print(f"E_exact - d_obj = {self.exact_sol - self.dual_objective}")
		
		
	def _initilize_vec(self, f_init = None):
		'''
		the vectors in the scs iteration are of the form u = [x, y, tao].
		this method initiates such variables. if f_init is not specified 
		x = zeros, y = identities / normalization and 
		tao = 1
		'''
		vec = np.zeros(self.len_joint_vec_u)
		
		if f_init is None:
			x = np.zeros(self.len_dual_vec_x)
			
			y = np.zeros(self.len_primal_vec_y)
			for pv in self.primal_vars:
				y[pv.slice] = (1 / pv.matdim)  * np.identity( pv.matdim ).ravel()
			
			tao = 1.0
			
		else:
			x = f_init((self.len_dual_vec_x, ))
					
			y = f_init((self.len_primal_vec_y, ) )
			
			tao = f_init((1,))
		
		vec[self.x_slice] = x
		vec[self.y_slice] = y
		vec[self.tao_slice] = tao
			
		return vec
	
	def _check_termination_criteria(self):
		''' 
		compute residuals as defined in 
		https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf (sec 3.5)
		and retrun True if all stopping criteria are satisfied
		'''
		sigma = self.sigma
		rho = self.rho
		
		# vars of RESCALED problem
		tao =self.u[self.tao_slice]
		x_k = self.u[self.x_slice] / tao
		s_k = self.v[self.y_slice] / tao
		y_k = self.u[self.y_slice] / tao
		
		# primal residuals =  2-norm(1/sigma*(Ax+s-b))
		self.resid_prim =  np.linalg.norm((1/sigma) * (apply_dual_constr(self, x_k) + s_k - self.b ) )
		
		# dual residuals =  2-norm(1/rho*(A.Ty + c))
		self.resid_dual = np.linalg.norm((1/rho) * (apply_primal_constr(self, y_k) + self.c ))
		
				# both objectives rescaled to original 
		cx = (1/(sigma*rho)) * np.vdot(self.c, x_k)
		by = (1/(sigma*rho)) * np.vdot(self.b, y_k)
		# gap residuals = abs(c.T*x + b.T*y) (already scaled back)
		self.resid_gap = abs(cx + by)
		
		# current primal objective (note the minus sign , see sign_convention_doc.txt)
		self.primal_objective = by
		# current dual objective
		self.dual_objective = -cx
		
		# average cg iterations since last time termination criteria checked
		self.cg_avrg_iter_count = self.cg_iter_counter / self.iter_counter_for_cg_avg
		self.cg_iter_counter = 0 
		self.iter_counter_for_cg_avg = 0
		
		if any((np.isnan(self.resid_prim), np.isnan(self.resid_dual), np.isnan(self.resid_gap))):
			logger.critical("residuals have nan values")
			raise Exception('numerical problem')
		
		 
		return all( [self.resid_prim < self.settings['scs_prim_resid_tol'] * (1.0 + self.b_unscld_norm), \
			self.resid_dual < self.settings['scs_dual_resid_tol'] * (1.0 + self.c_unscld_norm ), \
			self.resid_gap < self.settings['scs_gap_resid_tol'] * (1.0 + abs(cx) + abs(by)) ] )
		
		
		
	def _adapt_scaling(self):
		resid_ratio = self.resid_prim / self.resid_dual
		if resid_ratio > self.settings['scs_adapt_scale_if_ratio'] or resid_ratio < (1 / self.settings['scs_adapt_scale_if_ratio']):
			
			sigma = self.sigma
			rho = self.rho
			tao =self.u[self.tao_slice]
			x_k = self.u[self.x_slice] / tao / sigma
			s_k = self.v[self.y_slice] / tao / sigma
			y_k = self.u[self.y_slice] / tao / rho
			
			b = self.b / sigma
			c = self.c / rho
			
			rescale_factor = resid_ratio**(1/2)
			
			self.sigma = sigma * rescale_factor
			self.rho = rho #/ rescale_factor
			logger.info(f'rescaling: sigma = {sigma:0.3g} --> {self.sigma:0.3g}; rho = {rho:0.3g} --> {self.rho:0.3g}')
			self.b = b * self.sigma
			self.c = c * self.rho
			
			self.h[self.x_slice] = self.c
			self.h[self.y_slice] = self.b
			
			self.u[self.x_slice] = x_k * self.sigma * tao
			self.v[self.y_slice] = s_k * self.sigma * tao
			self.u[self.y_slice] = y_k * self.rho * tao
			
			self.Minv_h[self.x_slice], self.Minv_h[self.y_slice] = self.__solve_M_inv_return(self.c, self.b, tol = self.settings['cg_tol_Minvh_init'])
			self.hMinvh_plus_one_inv = 	1.0/(1.0 + np.vdot(self.h, self.Minv_h))
			
			
		
	
	def __iterate_scs_test(self):
		'''	 
		 see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 3.2.3 and 3.3
		 q is the over-relaxation parameter PD.q
		
		my properly working matlab function was:
		
			utilde= projectToAffine(u+v,iter,PD,FH);
			qcomb   = (PD.q*utilde + (1-PD.q)*u );
			u  = projectToCone(qcomb - v,PD);
			v = v - qcomb + u; 
		
		'''
		
		self._project_to_affine(self.u + self.v, out = self.u_tilde)
		qcomb = self.q * self.u_tilde + (1.-self.q) * self.u
		self._project_to_cone( qcomb - self.v, out = self.u);
		self.v += -qcomb + self.u ;
		
		
		return
	
	
	
	def _iterate_scs(self):
		'''	 
		 see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 3.2.3 and 3.3
		 q is the over-relaxation parameter PD.q
		
		my properly working matlab function was:
		
			utilde= projectToAffine(u+v,iter,PD,FH);
			qcomb   = (PD.q*utilde + (1-PD.q)*u );
			u  = projectToCone(qcomb - v,PD);
			v = v - qcomb + u; 
		
			rewrite as
			
			utilde = projectToAffine(u+v,iter,PD,FH);
			qcomb_minus_v   = PD.q*utilde + (1-PD.q)*u -v ;
			u = projectToCone(qcomb_minus_v,PD);
			v = u - qcomb_minus_v; 
			
			change sign of qcomb_minus_v
			
			utilde = projectToAffine(u+v,iter,PD,FH);
			v_minus_qcomb   = v - PD.q*utilde - (1-PD.q)*u  ;
			u = projectToCone( -1.* v_minus_qcomb,PD);
			v = u + v_minus_qcomb; 
			
		--> we can store the intermediate v_minus_qcomb in v:
		
		instead of 
		project_to_affine(u+v, out = u_tilde)
		q_comb = q * u_tilde + (1.-q) * u 
		project_to_cone(q_comb - v, out = u);
		v +=  u - qcomb; 
		
		we have:
		'''
		
		self._project_to_affine(self.u + self.v, out = self.u_tilde)
		self.v += -self.q * self.u_tilde + -(1.-self.q) * self.u # v = v- qcomb
		self._project_to_cone( -self.v, out = self.u);
		self.v +=  self.u ;
		
		self.iter_counter_for_cg_avg += 1
		return
	
	
 

	def _project_to_cone(self,u, out ):
		'''
		project each PSD variable to PSD cone 
		(diagonalize --> set negative eigs to 0 )
		'''
		
		"""
		 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
			for index in range(2):
				executor.submit(database.update, index)
		# .submit(function, *args, **kwargs)
		"""
	
		t = time.perf_counter()

		out[...] = u
		
		if self.settings['thread_multithread']:
			ctx_mgr = concurrent.futures.ThreadPoolExecutor(max_workers=self.settings["thread_max_workers"])
		else:
			ctx_mgr = util.NullContextManager(util.DummyExecutor()) # a dummy context
 
		with ctx_mgr as executor:
			for var_list,u_slice in zip((self.primal_vars, self.dual_vars), (self.y_slice, self.x_slice)) :
				for var in var_list:
					executor.submit(self._project_single_var_to_cone,  u=u, out=out, var=var, u_slice=u_slice )
					
		# project tao to R_+
		if u[self.tao_slice] < 0:
			out[self.tao_slice] = 0.
		
		t = time.perf_counter() - t
		self.t_cone += t


	def _project_single_var_to_cone(self, u, out, var, u_slice):
		if var.cone == 'PSD':  
			eigenvals, U =  self._eig_impl( np.reshape(u[u_slice][var.slice], (var.matdim, var.matdim) ) )
			# print(var.name, eigenvals)
			eigenvals[eigenvals < 0 ] = 0 # set negative eigenvalues to 0
			# print(var.name, eigenvals)
			
			out[u_slice][var.slice] = (U @ np.diag(eigenvals) @ U.conj().T).ravel() # and rotate back

	
	def _eig_impl(self, M):
		# return scipy_eigsh(M)
		return np.linalg.eigh(M)


	def _project_to_affine(self, w, out):
		''' 
		solves (I+Q)u=w
		acts in place, i.e. w <-- solution u
		
		see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf (4.1)
		
		previous matlab func:
			function [u,stats] = projectToAffine(w,iter,PD,FH)
			% PD is the problem data structure. this function adds the Minvh field and
			% others
			% if Minv*h is not cached compute it  
		 
			[xindsInU,yindsInU,taoindInU]=Uinds(PD);

			% h= [c;b];
			w_tao=w(taoindInU);
			w_xIN=w(xindsInU) -w_tao*(PD.c);
			w_yIN=w(yindsInU) -w_tao*(PD.b);

			[out_x,out_y,stats] = solveMinv(w_xIN ,w_yIN,PD,FH, PD.CGtolFunc(iter));
			out=[out_x;out_y];
			u_xy = out - PD.const_hMh * PD.Minvh * ([PD.c;PD.b]' * out);
			% ad
			u = [u_xy; w_tao + (PD.c)'*u_xy(xindsInU) + (PD.b)'*u_xy(yindsInU)];

			% stats.testSol=norm(w-eyePlusQ(u));
			end
		
		'''
		t = time.perf_counter()
		
		out[...] = w	
		out_tao = out[self.tao_slice]
		out_x = out[self.x_slice]
		out_y = out[self.y_slice]
		
		
		out_x += -w[self.tao_slice] * self.c
		out_y += -w[self.tao_slice] * self.b
		
		self._solve_M_inv(out_x, out_y)  # out_x,out_y <-- sol to M@[x,y] = [out_x,out_y]
		
		
		Minvh_x = self.Minv_h[self.x_slice]
		Minvh_y = self.Minv_h[self.y_slice]
		# from above docstr
		# u_xy = out - PD.const_hMh * PD.Minvh * ([PD.c;PD.b]' * out);
		dot_prod_hw = np.vdot(self.c, out_x) + np.vdot(self.b, out_y)
		
		out_x += -self.hMinvh_plus_one_inv * dot_prod_hw * Minvh_x  
		out_y += -self.hMinvh_plus_one_inv * dot_prod_hw * Minvh_y
		
		# from above docstr 
		# u = [u_xy; w_tao + (PD.c)'*u_xy(xindsInU) + (PD.b)'*u_xy(yindsInU)];
		out_tao += np.vdot(self.c, out_x) + np.vdot(self.b, out_y)
		
		t = time.perf_counter() - t
		self.t_affine += t
		return
		




	def __solve_M_inv_return(self, w_x, w_y, tol = None):
		'''
		this is for debugging. should give the same result as solve_M_inv(...)
		also used for first computation of M_inv(h)
		
		'''
			
		ATw_y = apply_primal_constr(self, w_y)
		z_x, exit_code = self._conj_grad_impl( w_x - ATw_y, tol = tol)
		logger.debug(f' in __solve_M_inv_return: CG exit code {exit_code}') 
		Az_x = apply_dual_constr(self, z_x)
		z_y = w_y + Az_x
		# print(f'cg exit code: {exit_code}')
		return z_x, z_y
		
	

	def _solve_M_inv(self, w_x, w_y):
		'''
		ACT IN PLACE: the solution is written back into w_x,w_y
		
		previous matlab func:
			function [z_x,z_y,stats] = solveMinv(w_x,w_y,PD,FH,tol,maxIter)
			% see (28) in https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf

			stats=struct('CGflag',[],'CGrelres',[],'CGiters',[],'T_CG',[]);
			if nargin < 5
				tol=PD.CGtol;
			end
			if nargin < 6
			   maxIter=PD.CGmaxIter;
			end
				
			ATw_y   = FH.ATransposed_funcHandle(full(w_y));
			ticCG=tic;
			[z_x,stats.CGflag,stats.CGrelres,stats.CGiters]     = pcg(FH.eyePlusATA_funcHandle,full(w_x - ATw_y),tol,maxIter);
			stats.T_CG=toc(ticCG);
			Az_x    = FH.A_funcHandle(z_x);
			z_y     = w_y + Az_x;
		
		
		'''
		
		apply_primal_constr(self, -w_y, out = w_x ) # w_x <-- w_x - A^T @ w_y
		w_x[...], exit_code = self._conj_grad_impl(w_x) # w_x stores the solution z_x
		# print(f'cg exit code: {exit_code}')
		
		apply_dual_constr(self, w_x, out = w_y) # w_y <-- w_y + A @ z_x : w_y stores the solution z_y
		
		return 0
		
			

	def _conj_grad_impl(self, x, tol = None):
		'''
		put your favourite iterative solver here
		'''		
		if tol is None:
			tol = self.cg_tol
			
		#  norm(b - A @ x) <= max(rtol*norm(b), atol)
		
		return scipy_cg(A = self.lin_op, b = x,\
			atol= self.settings['cg_atol'],\
			rtol = tol,\
			maxiter = self.settings['cg_maxiter'],\
			callback = self._cg_count_iter
		)
		
		
	def _cg_count_iter(self, x_k):
		'''
		callback function for conjugat gradient.
		simply counts iterations.
		'''
		self.cg_iter_counter += 1
		
	def __apply_M(self,x,y):
		'''
		function [Mxy] = applyM(xy,PD,FH)
		% M= [I,A^T; -A,I]
		% applyMsymm takes  [x;y] and produces [x +A^T*y; -Ax+y];
		 
		[xindsInU,yindsInU]=Uinds(PD);

		Mxy=nan(size(xy)); %initialize
		Mxy(xindsInU) = xy(xindsInU) + FH.ATransposed_funcHandle(xy(yindsInU));
		Mxy(yindsInU) = xy(yindsInU) - FH.A_funcHandle(xy(xindsInU));
		'''

		x_out =  x +  apply_primal_constr(self,y)
		y_out = y - apply_dual_constr(self,x)	
		 
		return x_out, y_out
		
		
	def __one_plus_Q(self, u):
		'''
		returns the result
		
		
		function [eyePlusQu] = eyePlusQ(u,PD,FH)
		% apply eye+Q (see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 4.1)
		% eye+Q = [M,h;-h^T,1] ie (eye+Q)[xy,tao] = [M*[xy]+h*tao ; -h^T*xy + tao]
		 

		[xindsInU,yindsInU,taoindInU]=Uinds(PD);
		xyindsInU=[xindsInU,yindsInU];
		 
		h=[PD.c;PD.b];

		eyePlusQu=nan(size(u)); %initialize
		eyePlusQu(xyindsInU) =  applyM(u(xyindsInU),PD,FH) + h*u(taoindInU);
		eyePlusQu(taoindInU) =  -h'*u(xyindsInU) + u(taoindInU);
		'''
		u_out = np.zeros(len(u))
		
		u_x = u[self.x_slice]
		u_y = u[self.y_slice]
		u_tao = u[self.tao_slice]
		
		# eyePlusQu(xyindsInU) =  applyM(u(xyindsInU),PD,FH) + h*u(taoindInU);
		u_out_x, u_out_y = self.__apply_M(u_x,u_y) 
		u_out_x += u_tao * self.c 
		u_out_y += u_tao * self.b
		
		# eyePlusQu(taoindInU) =  -h'*u(xyindsInU) + u(taoindInU);
		u_out_tao = np.vdot(-self.b, u_y) + np.vdot(-self.c, u_x) + u_tao
		
		u_out[self.x_slice] = u_out_x
		u_out[self.y_slice] = u_out_y
		u_out[self.tao_slice] = u_out_tao
		
		return u_out
		




	def __project_to_affine_return(self, w):
		''' 
		solves (I+Q)u=w
		returns the solution u
		
		see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf (4.1)
		
		previous matlab func:
			function [u,stats] = projectToAffine(w,iter,PD,FH)
			% PD is the problem data structure. this function adds the Minvh field and
			% others
			% if Minv*h is not cached compute it  
		 
			[xindsInU,yindsInU,taoindInU]=Uinds(PD);

			% h= [c;b];
			w_tao=w(taoindInU);
			w_xIN=w(xindsInU) -w_tao*(PD.c);
			w_yIN=w(yindsInU) -w_tao*(PD.b);

			[out_x,out_y,stats] = solveMinv(w_xIN ,w_yIN,PD,FH, PD.CGtolFunc(iter));
			out=[out_x;out_y];
			u_xy = out - PD.const_hMh * PD.Minvh * ([PD.c;PD.b]' * out);
			% ad
			u = [u_xy; w_tao + (PD.c)'*u_xy(xindsInU) + (PD.b)'*u_xy(yindsInU)];

			% stats.testSol=norm(w-eyePlusQ(u));
			end
		
		'''
		
			
		w_tao = w[self.tao_slice]
		w_x = w[self.x_slice]
		w_y = w[self.y_slice]
		
		z_x, z_y = self.__solve_M_inv_return(w_x - w_tao * self.c , w_y - w_tao * self.b)   
		
		
		Minvh_x = self.Minv_h[self.x_slice]
		Minvh_y = self.Minv_h[self.y_slice]
		# from above docstr
		# u_xy = out - PD.const_hMh * PD.Minvh * ([PD.c;PD.b]' * out);
		dot_prod_hz = np.vdot(self.c, z_x) + np.vdot(self.b, z_y)
		
		u_x = z_x - self.hMinvh_plus_one_inv * dot_prod_hz * Minvh_x  
		u_y = z_y - self.hMinvh_plus_one_inv * dot_prod_hz * Minvh_y
		
		# from above docstr 
		# u = [u_xy; w_tao + (PD.c)'*u_xy(xindsInU) + (PD.b)'*u_xy(yindsInU)];
		u_tao = w_tao +  np.vdot(self.c,u_x) + np.vdot(self.b,u_y)
		
		u = np.zeros(self.len_joint_vec_u)
		u[self.x_slice] = u_x
		u[self.y_slice] = u_y
		u[self.tao_slice] = u_tao
		return u
		



 
class LinOp_id_plus_AT_A(LinearOperator):
	'''
	linear operator with a buffer to store the intermediate y = Ax vector in the calculation of (1 + A^T @ A)x
	'''
	def __init__(self, len_x, len_y, dtype, primal_constraints, dual_constraints, settings):
		
		self.len_dual_vec_x = len_x
		self.len_primal_vec_y = len_y
		self.primal_constraints = primal_constraints
		self.dual_constraints = dual_constraints
		self.dtype = dtype
		self.settings = settings
		
		self.y_buffer = np.zeros(len_y, dtype = dtype)
		self.x_buffer = np.zeros(len_x, dtype = dtype)
		
		
		super().__init__(shape = (len_x, len_x), dtype = dtype    )
		
				
	def _matvec(self,x):
		'''
		implements x <-- x + A^T @ A @ x
		'''
		# # y_buffer <-- A @ x:
		# self.y_buffer = np.zeros(self.len_primal_vec_y, dtype = self.dtype)
		# self.x_buffer[...] = x
		# apply_dual_constr(self, x = x, out = self.y_buffer) 
		
		# # x += A^T @ y
		# apply_primal_constr(self, y = self.y_buffer, out = self.x_buffer)
		# return self.x_buffer
	
		# a bit slower:
		self.y_buffer = apply_dual_constr(self, x)
		return 	x + apply_primal_constr(self, self.y_buffer)

	
		
		
		
		
		

# the following functions are used by both the linear operator 1+A^\dagger*A and scs_solver.
# instead of making both instances of a parent calss which implements those functions as methods, 
# I chose to have them as separate functions which accept an object that has all the needed attributes.



def apply_primal_constr(obj, y, out = None ):
	''' 
	if out is specified then the function acts in place: v_out += primal_constraints(v_in)
	otherwise (out=None) it returns v_out = primal_constraints(v_in)
	'''
	return _impl_apply_constr(y, obj.primal_constraints, obj.settings, len_out = obj.len_dual_vec_x, out = out)

def apply_dual_constr(obj, x, out = None ):
	'''
	if out is specified then the function acts in place: v_out += dual_constraints(v_in)
	otherwise (out=None) it returns v_out = dual_constraints(v_in)
	'''
	# print('applying dual constraints', 'constraints list:', [c.label for c in obj.dual_constraints])
	# print('out var = ', out)
	return _impl_apply_constr(x, obj.dual_constraints, obj.settings, len_out = obj.len_primal_vec_y, out = out)



def _impl_apply_constr(v_in, constr_list, settings, out = None, len_out = None):
	''' 
	if out is specified then the function acts in place: v_out += constraints(v_in)
	otherwise (out=None) it returns the result
	'''
	if out is None:
		out = np.zeros(len_out)
		return_flag = True
	else:
		return_flag = False
	
	if settings['thread_multithread']:
		ctx_mgr = concurrent.futures.ThreadPoolExecutor(max_workers=settings["thread_max_workers"])
	else:
		ctx_mgr = util.NullContextManager(util.DummyExecutor()) # a dummy context
 
	with ctx_mgr as executor:
		for c in constr_list:
			executor.submit(c.__call__,  v_in=v_in, v_out = out )

	if return_flag:
		return out
	else:
		return None

 
	
 
