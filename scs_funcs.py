import scipy.linalg as la
import numpy as np
from variables import OptVar
import matrix_aux_functions as mf
from scipy.sparse.linalg import LinearOperator
from constraints import Constraint

import inspect 
 


def scs_iteration():
	return

def project_to_cone(v, primal_or_dual = 'primal'):
	'''
	project each PSD variable to PSD cone 
	(diagonalize --> set negative eigs to 0 )
	'''
	if primal_or_dual == 'primal':
		var_list = OptVar.primal_vars
	else:
		var_list = OptVar.dual_vars
		
	for var in var_list:
		if var.cone == 'PSD': # else if == 'Rn': projection to Rn = identity
					
			eigenvals, U =  la.eigh( np.reshape(v[var.slice], (var.matdim, var.matdim) ) )
			eigenvals[eigenvals < 0 ] = 0 # set negative eigenvalues to 0

			v[var.slice] = (U @ np.diag(eigenvals) @ U.conj().T).ravel() # and rotate back
	
	
	return 0


def apply_primal_constr(y, out, add_to_out = False):
	for c in Constraint.primal_constraints:
		c.__call__(v_in = y, v_out = out, add_to_out = add_to_out)
		

def apply_dual_constr(x, out, add_to_out = False):
	for c in Constraint.dual_constraints:
		c.__call__(v_in = x, v_out = out, add_to_out = add_to_out)


def _id_plus_AT_A(x, y_buffer):
    # y_buffer <-- A*x
    apply_dual_constr( x = x, out = y_buffer)
    # x += A^T*y
    apply_primal_constr( y = y_buffer, out = x, add_to_out = True)
    return x

 
		
		
		

class LinOp_id_plus_AT_A(LinearOperator):
	'''
	linear operator with a buffer to store the intermediate y = Ax vector in the calculation of (1 + A^T @ A)x
	'''
	def __init__(self, y_buffer = None):
		if not y_buffer:
			assert OptVar.lists_closed, \
				'''
				!!!!! need length of y vector to set buffer for linear op.\n
				variable lists in OptVar are not closed.\n
				run OptVar._close_var_lists() to close the lists and to fix OptVar.len_primal_vec_y
				'''
			
			self.y_buffer = np.zeros(OptVar.len_primal_vec_y, dtype = OptVar.dtype)
		else: 
			self.y_buffer = y_buffer
				
		super().__init__(shape = (OptVar.len_dual_vec_x,)*2, dtype = OptVar.dtype    )
	
	def _matvec(self,x):
		apply_dual_constr( x = x, out = self.y_buffer, add_to_out = False)
		# x += A^T*y
		apply_primal_constr( y = self.y_buffer, out = x, add_to_out = True)
		return x		





def project_to_affine(w):
	pass

# function [u,stats] = projectToAffine(w,iter,PD,FH)
	'''
	solves (I+Q)u=w
	see https://web.stanford.edu/~boyd/papers/pdf/scs_long.pdf (4.1)
	if Minv*h is not cached compute it  
	'''
	
 
'''
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
''' 



