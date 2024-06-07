import scipy.linalg as la
import numpy as np
from variables import OptVar
import matrix_aux_functions as mf

def project_to_cone(u):
	'''
	u = [x,y,tao]
	x consists of dual vars, y of primal
	'''

	tao = u[2]

	# project tao to R_{+}
	if tao <= 0:
		tao = 0. 

	# project each PSD variable in x and y to PSD cone
	for u_component, var_list in zip((u[0], u[1]), (OptVar.dual_vars, OptVar.primal_vars)):
		for var in var_list:
			if var.cone == 'PSD': # else if == 'Rn': projection to Rn = identity
				mat = mf.vec2mat( dim = np.prod(var.dims), vec = u_component[var.slice])
				
				eigenvals, U = la.eigh(mat)
				eigenvals[eigenvals < 0 ] = 0 # set negative eigenvalues to 0
				
				mat = U @ np.diag(eigenvals) @ U.conj().T # and rotate back
				
				u_component[var.slice] = mf.mat2vec( dim = np.prod(var.dims), mat = mat)
				



