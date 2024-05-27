
		
import numpy as np

def dim_symm_matrix(d):
	return int((d**2 - d)/2 +d)
	
def dim_AntiSymm_matrix(d):
	return int((d**2 - d)/2)
	
	


def vec2mat(dim, real_part, imag_part = None):
	'''
	combine vectors of real and imaginary parts into a hermitian matrix.
	convention is to store the upper triangular part in row major form 
	(= to lower triangular part in column major form)
	'''
		
	if not isinstance(real_part, np.ndarray):
		real_part = np.array(real_part, dtype = float)
	
	assert len(real_part) == dim_symm_matrix(dim), \
		f'real part has wrong number of entries: len(real) = {len(real_part)}, dimSymmMat = {dim_symm_matrix(dim)}'
	
	if imag_part:
		
		if not isinstance(imag_part, np.ndarray):
			imag_part = np.array(imag_part, dtype = float)
		
		assert len(imag_part) == dim_AntiSymm_matrix(dim), \
		f'imagenary part has wrong number of entries: len(imag) = {len(imag_part)}, dimAntiSymmMat = {dim_AntiSymm_matrix(dim)}'
		
	if imag_part:
		mat_out = np.zeros((dim,dim), dtype = np.cfloat )
	else:
		mat_out = np.zeros((dim,dim), dtype = np.double )
	
	if dim == 1:
		mat_out = real_part
	else:
		mat_out[np.triu_indices(dim)] += real_part
		mat_out					 -= np.diag(np.diag(mat_out))
		mat_out[np.tril_indices(dim)] += real_part
		
		if imag_part:
			imag_part = imag_part.astype(complex)
			mat_out[np.triu_indices(dim, k = 1)]  += 1.0j * imag_part
			mat_out[np.tril_indices(dim, k = -1)] -= 1.0j * imag_part
			
	return mat_out


def mat2vec(dim, matrix, complex_input = True):
	assert matrix.shape == (dim, dim), 'check simensions of input matrix'
	
	real_part = np.real(matrix(np.triu_indices(dim)))
	
	if complex_input:
		imag_part = np.imaginary(matrix(np.triu_indices(dim), k = 1))
	else:
		imag_part = None
	
	return (real_part, imag_part)


rp = [1,2,3]
ip = [4]

A = vec2mat(2, rp, ip)
print(A)
