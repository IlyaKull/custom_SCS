
		
import numpy as np

def dim_symm_matrix(d):
	return int((d**2 - d)/2 +d)
	
def dim_AntiSymm_matrix(d):
	return int((d**2 - d)/2)
	
	


def vec2mat(dim, real_part, imag_part = None, checks = False):
	'''
	combine vectors of real and imaginary parts into a hermitian matrix.
	convention is to store the upper triangular part in row major form 
	(= to lower triangular part in column major form)
	'''
	if checks: # validate input. will be switched off
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
			# imag_part = imag_part.astype(complex)
			mat_out[np.triu_indices(dim, k = 1)]  += 1.0j * imag_part
			mat_out[np.tril_indices(dim, k = -1)] -= 1.0j * imag_part
			
	return mat_out


def mat2vec(dim, matrix, complex_input = True):
	assert matrix.shape == (dim, dim), 'check simensions of input matrix'
	
	real_part = np.real(matrix[np.triu_indices(dim)])
	
	if complex_input:
		imag_part = np.imag(matrix[np.triu_indices(dim, k = 1)])
	else:
		imag_part = None
	
	return (real_part, imag_part)



'''
################ Testing vec2mat

rp = np.array([1,2,3])
ip = np.array([4])

A = vec2mat(2, rp, ip, checks=False)
print(A)

RP, IP = mat2vec(2, A)
print(f'real part {RP}\n imag part {IP}')


'''



def partial_trace(x, subsystems, dims, checks = False):
	
	if checks:
		assert isinstance(subsystems, list) or isinstance(subsystems, tuple) , 'subsystems should be a list or a tuple'
		assert max(subsystems) <= len(dims)
		
	dim_out = np.prod( [d for i,d in enumerate(dims,1) if i not in subsystems])
	
	
	inds= list(range(1,len(dims+dims)+1))  # dims = [2,3,2] ---> inds= [1,2,3,4,5,6]
	inds[subsystems + [i+len(dims) for i in subsystems] ] = 0 #  subsystems = [3] ---> inds= [ 1,2,0,4,5,0]
	return np.reshape(
			np.einsum( x.reshape(dims + dims), inds, [i for i in inds if i != 0]),  # [ 1,2,0,4,5,0] --> [1,2,4,5] i.e. 'ijklmk-->ijlm'
			(dim_out,dim_out) 
			)






