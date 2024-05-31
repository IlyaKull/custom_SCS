
		
import numpy as np
import functools as ft


def tensorProd(*terms):
	'''
	admits list input as well as *args
	'''

	if len(terms)==1 and isinstance(terms[0], list):
		terms = terms[0]
		
	return ft.reduce(np.kron, terms)
		



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
		assert len(dims) <= 26, 'np.einsum cannot handle more than 52 indices  =[' 
		
	
	kept_sys = [ (i not in subsystems) for i in range(1,len(dims)+1)]
	
	dim_out = np.prod(np.array(dims)[kept_sys])
		
	# inds for np.einsum
	inds_in = np.arange(1,2*len(dims)+1).reshape((2,len(dims)))  # dims = [2,3,2] ---> inds_in= [1,2,3; 4,5,6]
	subsys_inds = np.array(subsystems)-1
	inds_in[1,subsys_inds] = inds_in[0,subsys_inds] # subsystems = [2,3] ---> inds_in = [1,2,3; 4,2,3]
	inds_out  = inds_in[:,np.array(kept_sys)] # [1,2,3; 4,2,3] --> [1;4] i.e. 'ijkljk-->il'
	
		
	return np.reshape(
			np.einsum( x.reshape(dims + dims), inds_in.flatten().tolist(), inds_out.flatten().tolist()), 
			(dim_out,dim_out) 
			)



# # '''
# # test partial_trace
# # '''

# # X = np.array([[0,1],[1,0]])
# # A = np.array([[1,0],[0,3]])
# # I = np.identity(2)
# # AX = np.kron(A,X)
# # AXX= np.kron(AX,X)
# # XXA = np.kron(X,np.kron(X,A))


# # # p1AX = partial_trace(AX, [1], [2,2])
# # # print(p1AX)

# # # p2AX = partial_trace(AX, [2], [2,2])
# # # print(p2AX)

# # # p1AXX = partial_trace(AXX, [1], [2,2,2])
# # # print(p1AXX)

# # # p2AXX = partial_trace(AXX, [2], [2,2,2])
# # # print(p2AXX)

# # # p3AXX = partial_trace(AXX, [3], [2,2,2])
# # # print(p3AXX)

# # B = np.arange(9).reshape((3,3))

# # print(np.allclose( partial_trace(tensorProd(B,A,I), [3], [3,2,2]), np.trace(I) * tensorProd(B,A)))
# # print(np.allclose( partial_trace(tensorProd(B,A,I), [1], [3,2,2]), np.trace(B) * tensorProd(A,I)))
# # print(np.allclose( partial_trace(tensorProd(B,A,I), [2], [3,2,2]), np.trace(A) * tensorProd(B,I)))




def xOtimesI(x, subsystems, fulldims, checks = False):
	''' 
	adjoint of partial trace: adds \otimes \id terms in specified subsystems
	'''
	kept_sys = [ (i not in subsystems) for i in range(1,len(fulldims)+1)]
	
	kept_dims = [fulldims[i] for i in range(len(fulldims)) if kept_sys[i]]
	traced_dims = [fulldims[i] for i in range(len(fulldims)) if not kept_sys[i]]
	 
	if checks:
		assert x.shape[0] == x.shape[1] and x.shape[0] == np.prod(np.array(fulldims)[kept_sys]),\
			f"x has wrong dimensions: x.shape = {x.shape}, dimsfull= {dimsfull}, subsystems= {subsystems}"
	
	# id terms are added on the right and then permuted to their place 
	xI = np.kron(x,\
			np.identity(\
				np.prod(\
					[ d for i,d in enumerate(fulldims) if not kept_sys[i]]
				)
			)
		).reshape( kept_dims + traced_dims + kept_dims + traced_dims )
		
	
	perm_l = np.zeros((len(fulldims)),dtype = int)
	perm_l[kept_sys] = range(sum(kept_sys))
	perm_l[[ not b for b in kept_sys] ] = range(sum(kept_sys),len(fulldims))
	perm_r = np.zeros((len(fulldims)),dtype = int)
	perm_r[kept_sys] = [ i+ len(fulldims) for i in range(sum(kept_sys))]
	perm_r[[ not b for b in kept_sys] ] = [ i+ len(fulldims) for i in range(sum(kept_sys),len(fulldims)) ]
	
	perm  = perm_l.tolist() + perm_r.tolist()
	
	if checks: 
		assert fulldims + fulldims == [(kept_dims + traced_dims + kept_dims + traced_dims)[i] for i in perm ], 'permutation check!!'
	
	return np.transpose(xI, axes = perm).reshape( (np.prod(fulldims),np.prod(fulldims)) )
	

# # # '''
# # # test xOtimesI
# # # '''	
 
 
# # # X = np.array([[0,1],[1,0]])
# # # A = np.arange(16).reshape((4,4))
# # # B = np.arange(4).reshape((2,2))
# # # I2 = np.identity(2)
# # # I3 = np.identity(3)

# # # print('I3 =\n',I3)

# # # AX = np.kron(A,X)
# # # BX = np.kron(B,X)

	

# # # print(np.allclose( xOtimesI(AX, [2], [4,2,2]) , tensorProd(A,I2,X)))
# # # print(np.allclose( xOtimesI(AX, [3], [4,2,3]) , tensorProd(A,X,I3)))
# # # print(np.allclose( xOtimesI(AX, [3], [4,2,3]) , tensorProd(A,X,I3)))


 
# # # print(np.allclose( xOtimesI(BX, [1], [3,2,2],checks=True) , tensorProd(I3,B,X))) # !!!!!!!!!!!!!!!!!!!!!!!!
# # # # print('BX =\n',BX)
# # # # print(tensorProd(I3,B,X))
# # # # print(xOtimesI(BX, [1], [3,2,2]))
 


