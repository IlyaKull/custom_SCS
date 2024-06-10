import scipy.linalg as la
import numpy as np
from variables import OptVar
import matrix_aux_functions as mf
from scipy.sparse.linalg import LinearOperator
from constraints import Constraint


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
					
			eigenvals, U =  la.eigh( np.reshape(v[var.slice], (np.prod(var.dims),)*2 ) )
			eigenvals[eigenvals < 0 ] = 0 # set negative eigenvalues to 0

			v[var.slice] = (U @ np.diag(eigenvals) @ U.conj().T).ravel() # and rotate back
	
	
	return 0


def apply_primal_constr(y, out = None, add_to_out = False):
	if out:
		x = out
	else: # allocate fresh array
		x = np.zeros( OptVar.dual_vars[-1].slice.stop, dtype = OptVar.dual_vars[-1].dtype)
	
	for c in Constraint.primal_constraints:
		c.__call__(v_in = y, v_out = x, add_to_out = add_to_out)
		


def apply_dual_constr(x, out = None, add_to_out = False):
	if out:
		y = out
	else: # allocate fresh array
		y = np.zeros( OptVar.primal_vars[-1].slice.stop, dtype = OptVar.primal_vars[-1].dtype)
	
	for c in Constraint.dual_constraints:
		c.__call__(v_in = x, v_out = y, add_to_out = add_to_out)


def id_plus_AT_A(x, y_buffer):
    # y_buffer <-- A*x
    apply_dual_constr( v_in = x, v_out = y_buffer)
    # x += A^T*y
    apply_primal_constr( v_in = y_buffer, v_out = x, add_to_out = True)
    return x

A = LinearOperator((dimY,dimX), matvec=id_plus_AT_A) ################# to do subclass to pass params!!!!!!!!!!!!!!!!!!







def project_to_affine(w)
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



