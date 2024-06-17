import scipy.linalg as la
import numpy as np
from variables import OptVar
import matrix_aux_functions as mf
from scipy.sparse.linalg import LinearOperator, cg
from constraints import Constraint

import inspect 
 


def scs_iteration(u,v,u_tilde):
	'''	 
	 see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf 3.2.3 and 3.3
	 q is the over-relaxation parameter PD.q
	
	my properly workin matlab function was:
	
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
	
	project_to_affine(u+v, out = u_tilde)
	v += -q * u_tilde + -(1.-q) * u # v = v- qcomb
	project_to_cone( -v, out = u);
	v +=  u ;

	return

def project_to_cone(u, out ):
	'''
	project each PSD variable to PSD cone 
	(diagonalize --> set negative eigs to 0 )
	'''
	out[...] = u
	
	for var_list,u_slice in zip((OptVar.primal_vars, OptVar.dual_vars), (OptVar.y_slice, OptVar.x_slice)) :
		for var in var_list:
			if var.cone == 'PSD':  
				eigenvals, U =  la.eigh( np.reshape(u[u_slice][var.slice], (var.matdim, var.matdim) ) )
				eigenvals[eigenvals < 0 ] = 0 # set negative eigenvalues to 0

				out[u_slice][var.slice] = (U @ np.diag(eigenvals) @ U.conj().T).ravel() # and rotate back
			
	# project tao to R_+
	if u[OptVar.tao_slice] < 0:
		out[OptVar.tao_slice] = 0.
	
	return 0

@profile
def apply_primal_constr(y, out = None ):
	''' 
	if out is specified then the function acts in place: v_out += primal_constraints(v_in)
	otherwise (out=None) it returns v_out = primal_constraints(v_in)
	'''
	if not out is None:
		for c in Constraint.primal_constraints:
			c.__call__(v_in = y, v_out = out )
		
		return 0
	else:
		out = np.zeros(OptVar.len_dual_vec_x)
		for c in Constraint.primal_constraints:
			c.__call__(v_in = y, v_out = out )
		
		return out 
@profile
def apply_dual_constr(x, out = None ):
	'''
	if out is specified then the function acts in place: v_out += dual_constraints(v_in)
	otherwise (out=None) it returns v_out = dual_constraints(v_in)
	'''
	if not out is None:
		for c in Constraint.dual_constraints:
			c.__call__(v_in = x, v_out = out )
		
		return 0
	else:
		out = np.zeros(OptVar.len_primal_vec_y)
		for c in Constraint.dual_constraints:
			c.__call__(v_in = x ,v_out = out  )
		
		return out

# # def _id_plus_AT_A(x, y_buffer):
    # # # y_buffer <-- A*x
    # # apply_dual_constr( x = x, out = y_buffer)
    # # # x += A^T*y
    # # apply_primal_constr( y = y_buffer, out = x, add_to_out = True)
    # # return x

 
def _id_plus_AT_A(x):
		return 	x + apply_primal_constr(apply_dual_constr(x))
		
		

 
class LinOp_id_plus_AT_A(LinearOperator):
	'''
	linear operator with a buffer to store the intermediate y = Ax vector in the calculation of (1 + A^T @ A)x
	'''
	def __init__(self):
		 
		self.y_buffer = np.zeros(OptVar.len_primal_vec_y, dtype = OptVar.dtype)
		self.x_buffer = np.zeros(OptVar.len_dual_vec_x, dtype = OptVar.dtype)
		
		super().__init__(shape = (OptVar.len_dual_vec_x,)*2, dtype = OptVar.dtype    )
		
		
		
	def _matvec(self,x):
		'''
		implements x <-- x + A^T @ A @ x
		'''
		
		# y_buffer <-- A @ x:
		self.y_buffer[...] = np.zeros(OptVar.len_primal_vec_y, dtype = OptVar.dtype)
		self.x_buffer[...] = x
		apply_dual_constr( x = x, out = self.y_buffer) 
		
		# x += A^T @ y
		apply_primal_constr( y = self.y_buffer, out = self.x_buffer)
		return self.x_buffer
	
 
 

 

 

def _solve_M_inv_return(w_x,w_y,lin_op):
	'''
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
	
	ATw_y = apply_primal_constr(w_y)
	z_x, exit_code = cg(lin_op, w_x - ATw_y, atol= 1e-8, tol= 1e-8, maxiter = 1000, 
		# callback = lambda x: print(f"current iter: \n{x[:10]}")
		)
	Az_x = apply_dual_constr(z_x)
	z_y = w_y + Az_x
	print(f'cg exit code: {exit_code}')
	return z_x, z_y


@profile
def solve_M_inv(w_x,w_y,lin_op):
	'''
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
	
	here the solution is written back into w_x,w_y
	'''
	
	apply_primal_constr(-w_y, out = w_x ) # w_x <-- w_x - A^T @ w_y
	w_x[...], exit_code = cg(lin_op, w_x, atol= 1e-8, tol= 1e-8, maxiter = 1000) # w_x stores the solution z_x
	print(f'cg exit code: {exit_code}')
	
	apply_dual_constr(w_x, out = w_y) # w_y <-- w_y + A @ z_x : w_y stores the solution z_y
	
	return 0
	
	
	
	
def _apply_M(x,y):
	'''
	function [Mxy] = applyM(xy,PD,FH)
	% M= [I,A^T; -A,I]
	% applyMsymm takes  [x;y] and produces [x +A^T*y; -Ax+y];
	 
	[xindsInU,yindsInU]=Uinds(PD);

	Mxy=nan(size(xy)); %initialize
	Mxy(xindsInU) = xy(xindsInU) + FH.ATransposed_funcHandle(xy(yindsInU));
	Mxy(yindsInU) = xy(yindsInU) - FH.A_funcHandle(xy(xindsInU));
	'''

	x_out = x.copy()
	y_out = y.copy()
	x_out += apply_primal_constr(y)
	y_out += - apply_dual_constr(x)	
	 
	return x_out, y_out
	
	


def _one_plus_Q(u,c,b):
	'''
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
	
	u_x = u[OptVar.x_slice]
	u_y = u[OptVar.y_slice]
	u_tao = u[OptVar.tao_slice]
	
	# eyePlusQu(xyindsInU) =  applyM(u(xyindsInU),PD,FH) + h*u(taoindInU);
	u_out_x, u_out_y = _apply_M(u_x,u_y) 
	u_out_x += u_tao * c 
	u_out_y += u_tao * b
	
	# eyePlusQu(taoindInU) =  -h'*u(xyindsInU) + u(taoindInU);
	u_out_tao = np.vdot(-b,u_y) + np.vdot(-c,u_x) + u_tao
	
	u_out[OptVar.x_slice] = u_out_x
	u_out[OptVar.y_slice] = u_out_y
	u_out[OptVar.tao_slice] = u_out_tao
	
	return u_out
	



def project_to_affine(w,out, lin_op,  c,  b, hMh, Minvh):
	''' 
	previous matlab func:
		function [u,stats] = projectToAffine(w,iter,PD,FH)
		% solves (I+Q)u=w
		% see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf (4.1)
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
	if test_result:
		w_in = w.copy()
		
	w_tao = w[OptVar.tao_slice]
	w_x = w[OptVar.x_slice]
	w_y = w[OptVar.y_slice]
	
	
	w_x += -w_tao * c
	w_y += -w_tao * b
	
	solve_M_inv(w_x,w_y)  # w_x,w_y <-- sol to M@[x,y] = [w_x,w_y]
	
	
	Minvh_x = Minvh[OptVar.x_slice]
	Minvh_y = Minvh[OptVar.y_slice]
	# from above docstr
	 # u_xy = out - PD.const_hMh * PD.Minvh * ([PD.c;PD.b]' * out);
	dot_prod = np.vdot(c,w_x) + np.vdot(b,w_y)
	w_x += -hMh * dot_prod * Minvh_x  
	w_y += -hMh * dot_prod * Minvh_y
	
	# from above docstr 
	# u = [u_xy; w_tao + (PD.c)'*u_xy(xindsInU) + (PD.b)'*u_xy(yindsInU)];
	w_tao += np.vdot(c,w_x) + np.vdot(b,w_y)
	
	if test_result:
		print(f'In solveing (1+Q)u = w max abs difference between (1+Q)@sol and w_in = {max(abs(w_in - _one_plus_Q(w,c,b)))}')
	return
	



