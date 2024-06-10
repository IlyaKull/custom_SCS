import scipy.linalg as la
import numpy as np
from variables import OptVar
import matrix_aux_functions as mf
from scipy.sparse.linalg import LinearOperator

def scs_iteration():
	return

def project_to_cone(var_list):
	'''
	project each PSD variable to PSD cone 
	(diagonalize --> set negative eigs to 0 )
	'''
	for var in var_list:
		if var.cone == 'PSD': # else if == 'Rn': projection to Rn = identity
					
			eigenvals, U = la.eigh(var.matrix)
			eigenvals[eigenvals < 0 ] = 0 # set negative eigenvalues to 0

			var.matrix[...] = U @ np.diag(eigenvals) @ U.conj().T # and rotate back
	
	

	return 0





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


def mv(v):
    return np.array([2*v[0], 3*v[1]])

A = LinearOperator((2,2), matvec=mv)
A
