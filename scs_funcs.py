import numpy as np
from variables import OptVar
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg as scipy_cg
from constraints import Constraint
import time

class SCS_Solver:
	'''
	after defining all the optimization variables (OptVar class),
	and all the constraints on them (expressions involving Maps acting on OptVars)
	the problem has been specified.
	
	At this point the SCS_Solver class can be instantiated. 
	It contains all the methods and objects needed to perform the scs algorithm.
	https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 
	
	
	When initialized this class first  closes the variable lists in the OptVar calss and
	 retrieves all the calss variables from OptVar and Constraint classes, 
	 such that all the data neede for the algorithm is encapsulated in this object. 
	
	variable names based on 
	https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 3.2.3 and 3.3
	
	'''
	def __init__(self, settings = dict()):
		
		if not OptVar.lists_closed:  # closing the lists fixes the sizes of the primal and dual vectors
			OptVar._close_var_lists()
			
		
		self.settings = default_settings()  
		self.settings.update(settings)
		# print(self.settings)
		
		self.primal_constraints = Constraint.primal_constraints
		self.dual_constraints = Constraint.dual_constraints
		
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
		
		# the linear operator 1+A^\dagger *A 
		self.lin_op = LinOp_id_plus_AT_A(self.len_dual_vec_x,\
			self.len_primal_vec_y,\
			self.dtype,\
			self.primal_constraints,\
			self.dual_constraints\
		)
		
		# the vector h is defined as h = [c,b]
		# where b is the vector of constants in the constraint Ax+s=b, s>= 0
		# and c is the objective function: min(c@x)
		#### EXPLAIN THE SIGN CONVENTION HERE
		self.h = np.zeros(self.len_dual_vec_x + self.len_primal_vec_y)
		self.c = self.h[self.x_slice] # slices for convenienc
		self.b = self.h[self.y_slice]
		for cstr in self.primal_constraints:
			self.c[cstr.conjugateVar.slice] = cstr.const
		for cstr in self.dual_constraints:
			self.b[cstr.conjugateVar.slice] = cstr.const
		
		# needed for termination criteria
		self.b_norm = np.linalg.norm(self.b)
		self.c_norm = np.linalg.norm(self.c)
		# print(self.h.base is None) 
		# print(self.b.base is None)
				
		# M^(-1)h is needed in every iteration and is therefore computed
		# at initialization 
		Minv_h = np.zeros(self.len_dual_vec_x + self.len_primal_vec_y)
		Minv_h[self.x_slice], Minv_h[self.y_slice] = self.__solve_M_inv_return(self.c, self.b)
		
		# the following is also needed in every iteration	
		self.hMinvh_plus_one_inv = 	1.0/(1.0 + np.vdot(self.h, Minv_h))
		self.Minv_h = Minv_h
		
		# progress log tracks the following 
		self.resid_prim = None
		self.resid_dual = None
		self.resid_gap = None
		self.primal_objective = None
		self.dual_objective = None
		
		
	
	def run_scs(self, num_iters = None):
		
		if not num_iters is None:
			self.settings['scs_maxiter'] = num_iters
		
		maxiter = self.settings['scs_maxiter']
		self.iter = 0
		termination_criteria_satisfied = False
		self.t_start = time.perf_counter()
		
		self._print_log(print_head = True, print_line = False)
		
		while (self.iter < maxiter) and (not termination_criteria_satisfied):
			self._iterate_scs()
			self.iter += 1
			
			if self.iter % self.settings['scs_compute_resid_every_n_iters'] == 0:
				termination_criteria_satisfied = self._check_termination_criteria()
				self._print_log()
 	
	
	def _print_log(self, print_head = False, print_line = True):
				
		col_width = self.settings['log_col_width']
		 
		res_format_str = '0.8g'
		obj_format_str = '0.8g'
		
		log_columns = {"Iter" : 	{'val' : self.iter, 			'formt_str' : 'g'},
			"Prim res" : 			{'val' : self.resid_prim, 		'formt_str' : res_format_str},
			"Dual res" : 			{'val' : self.resid_dual, 		'formt_str' : res_format_str},
			"Gap res" : 			{'val' : self.resid_gap, 		'formt_str' : res_format_str},
			"Prim obj" :			{'val' : self.primal_objective, 'formt_str' : obj_format_str},
			"Dual obj" : 			{'val' : self.dual_objective, 	'formt_str' : obj_format_str},
			"Time": 				{'val' : time.perf_counter() - self.t_start, 'formt_str' : '0.3g'}
		}
		
		if print_head:
			print("="*72)
			print("STARTING SCS")
			print("="*72)

			for k in log_columns:
				print(k.ljust(col_width), end = " | " )
			print('')
			
		if print_line:
			for dikt in log_columns.values():
				print(format(dikt['val'], dikt['format_str']).ljust(col_width), end = " | " )
			print('')
			
		
		
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
		tao =self.u[self.tao_slice]
		x_k = self.u[self.x_slice] / tao
		s_k = self.v[self.y_slice] / tao
		y_k = self.u[self.y_slice] / tao
		
		# primal residuals =  2-norm(Ax+s-b)
		self.resid_prim = np.linalg.norm( apply_dual_constr(self, x_k) + s_k - self.b )
		
		# dual residuals = 2-norm(A.Ty + c)
		self.resid_dual = np.linalg.norm( apply_primal_constr(self, y_k) + self.c )
		
		# gap residuals = abs(c.T*x + b.T*y)
		cx = np.vdot(self.c, x_k)
		by = np.vdot(self.b, y_k)
		self.resid_gap = abs(cx + by)
		
		# current primal objective (note the minus sign , see sign_convention_doc.txt)
		self.primal_objective = by
		# current dual objective
		self.dual_objective = -cx
		
		return all( [self.resid_prim < self.settings['scs_prim_resid_tol'] * (1.0 + slef.b_norm), \
			self.resid_dual < self.settings['scs_dual_resid_tol'] * (1.0 + self.c_norm ). \
			self.resid_gap < self.settings['scs_gap_resid_tol'] < (1.0 + abs(cx) + abs(by)) ] )
	
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

		return

 

	def _project_to_cone(self,u, out ):
		'''
		project each PSD variable to PSD cone 
		(diagonalize --> set negative eigs to 0 )
		'''
		out[...] = u
		
		for var_list,u_slice in zip((self.primal_vars, self.dual_vars), (self.y_slice, self.x_slice)) :
			for var in var_list:
				if var.cone == 'PSD':  
					eigenvals, U =  la.eigh( np.reshape(u[u_slice][var.slice], (var.matdim, var.matdim) ) )
					eigenvals[eigenvals < 0 ] = 0 # set negative eigenvalues to 0

					out[u_slice][var.slice] = (U @ np.diag(eigenvals) @ U.conj().T).ravel() # and rotate back
				
		# project tao to R_+
		if u[self.tao_slice] < 0:
			out[self.tao_slice] = 0.
		
		return 0

		


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
		
		return
		




	def __solve_M_inv_return(self, w_x, w_y):
		'''
		this is for debugging. should give the same result as solve_M_inv(...)
		
		'''
		
		ATw_y = apply_primal_constr(self, w_y)
		z_x, exit_code = self._conj_grad_impl( w_x - ATw_y )
		Az_x = apply_dual_constr(self, z_x)
		z_y = w_y + Az_x
		print(f'cg exit code: {exit_code}')
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
		print(f'cg exit code: {exit_code}')
		
		apply_dual_constr(self, w_x, out = w_y) # w_y <-- w_y + A @ z_x : w_y stores the solution z_y
		
		return 0
		
	
	

	def _conj_grad_impl(self, x):
		'''
		put your favourite implementation of conjugate gradient here
		'''		
		# return scipy_cg(A = self.lin_op, b = x,\
			# atol= 1e-8,\
			# tol= 1e-8,\
			# maxiter = 1000 \
		# )
		return scipy_cg(A = self.lin_op, b = x,\
			atol= self.settings['cg_atol'],\
			tol= self.settings['cg_tol'],\
			maxiter = self.settings['cg_maxiter'] \
		)
		
	
		
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
	def __init__(self, len_x, len_y, dtype, primal_constraints, dual_constraints):
		
		self.len_dual_vec_x = len_x
		self.len_primal_vec_y = len_y
		self.primal_constraints = primal_constraints
		self.dual_constraints = dual_constraints
		self.dtype = dtype
		
		self.y_buffer = np.zeros(len_y, dtype = dtype)
		self.x_buffer = np.zeros(len_x, dtype = dtype)
		
		
		super().__init__(shape = (len_x, len_x), dtype = dtype    )
		
				
	def _matvec(self,x):
		'''
		implements x <-- x + A^T @ A @ x
		'''
		# y_buffer <-- A @ x:
		self.y_buffer = np.zeros(self.len_primal_vec_y, dtype = self.dtype)
		self.x_buffer[...] = x
		apply_dual_constr(self, x = x, out = self.y_buffer) 
		
		# x += A^T @ y
		apply_primal_constr(self, y = self.y_buffer, out = self.x_buffer)
		return self.x_buffer
	
		# a bit slower:
		# self.y_buffer = apply_dual_constr(x)
		# return 	x + apply_primal_constr(self.y_buffer)




# the following functions are used by both the linear operator 1+A^\dagger*A and scs_solver.
# instead of making both instances of a parent calss which implements those functions as methods, 
# I chose to have them as separate functions which accept an object that has all the needed attributes.



def apply_primal_constr(obj, y, out = None ):
	''' 
	if out is specified then the function acts in place: v_out += primal_constraints(v_in)
	otherwise (out=None) it returns v_out = primal_constraints(v_in)
	'''
	return _impl_apply_constr(y, obj.primal_constraints, len_out = obj.len_dual_vec_x, out = out)

def apply_dual_constr(obj, x, out = None ):
	'''
	if out is specified then the function acts in place: v_out += dual_constraints(v_in)
	otherwise (out=None) it returns v_out = dual_constraints(v_in)
	'''
	return _impl_apply_constr(x, obj.dual_constraints, len_out = obj.len_primal_vec_y, out = out)



def _impl_apply_constr(v_in, constr_list, out = None, len_out = None):
	''' 
	if out is specified then the function acts in place: v_out += constraints(v_in)
	otherwise (out=None) it returns the result
	'''
	if out is None:
		out = np.zeros(len_out)
		for c in constr_list:
			c.__call__(v_in, v_out = out )
		return out 
	else:
		for c in constr_list:
			c.__call__(v_in, v_out = out )
		
		return None

 
 


def default_settings():
	d = {
		'cg_atol' : 1e-8,
		'cg_tol' : 1e-8,
		'cg_maxiter' : 2000,
		'scs_maxiter': 2000,
		'scs_q' : 1.5,
		'scs_sigma' : 1, # not implemented yet
		'scs_rho' : 1,   # not implemented yet
		'scs_scaling_D': None, # not implemented yet
		'scs_scaling_E': None, # not implemented yet
		'scs_prim_resid_tol' : 1e-6,
		'scs_dual_resid_tol' : 1e-6,
		'scs_gap_resid_tol' : 1e-6,
		'scs_compute_resid_every_n_iters' : 100,
		'log_col_width' : 12, 
		}
	
	return d
	
