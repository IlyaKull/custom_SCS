import itertools
import scipy.sparse as sps		
import numpy as np
import functools as ft

import line_profiler



def main():
	
	
	test_partial_trace(partial_trace)	 
	test_xOtimesI(xOtimesI, N=10)
	
	
	for dims in [[2,2,2],[2,100,2],[5,2,5],[5,2,10,2,5]]:
		profile = line_profiler.LineProfiler()
		profile.enable_by_count()
		# profile.add_function(partial_trace_no_inds)
		# profile.add_function(partial_trace_inds)
		profile.add_function(xOtimesI_bc_multi_no_inds)
		profile.add_function(xOtimesI_LR)
		
		bench_xOtimes_LR(dims =dims, N=10)
		
		print(f"========================== dims = {dims} ================================")
		profile.print_stats()


def anticomm(A,B):
	return A@B + B@A



def tensorProd(*terms):
	'''
	admits list input as well as *args
	'''

	if len(terms)==1 and isinstance(terms[0], list):
		terms = terms[0]
		
	return ft.reduce(np.kron, terms)
		



def dim_symm_matrix(d):
	return int((d**2 - d)/2 +d)
	
# # # # def dim_AntiSymm_matrix(d):
	# # # # return int((d**2 - d)/2)
	


# # # # # # # def vec2mat(dim, vec):
	# # # # # # # '''
	# # # # # # # convention is to store the upper triangular part in row major form 
	# # # # # # # (= to lower triangular part in column major form)
	# # # # # # # '''
	# # # # # # # assert len(vec) == dim_symm_matrix(dim), \
		# # # # # # # f'vec has wrong number of entries: len(vec) = {len(vec)}, dimSymmMat = {dim_symm_matrix(dim)}'
	
	
	# # # # # # # mat = np.zeros((dim,dim), dtype = vec.dtype )
	
	# # # # # # # mat[np.triu_indices(dim)] += vec
	# # # # # # # mat += mat.T.conj()
	# # # # # # # mat[np.diag_indices(dim)] /= 2.0
	
				
	# # # # # # # return mat




# # # # # # # def mat2vec(dim, mat):
	# # # # # # # assert mat.shape == (dim, dim), 'check simensions of input matrix'
	
	# # # # # # # return mat[np.triu_indices(dim)]
	
	 


 
# ################ Testing vec2mat

# vec = np.arange(10.)
 
# print(f'vec = \n{vec}') 

# print(f'mat = \n{vec2mat(4, vec)}')

# print(f'vecFmat = \n{mat2vec(4,vec2mat(4, vec))}')
 



def partial_trace(x, subsystems, dims):
	'''
	the _einsum function is faster than the _bc function 
	'''
	
	if {s for s in subsystems} == {s+1 for s in range(len(dims))}:
		return np.array([[np.trace(x)]]) # return a 1x1 matrix
	elif subsystems == []:
		return x
	else:
		return partial_trace_no_inds(x, dims, *partial_trace_inds(subsystems, dims))
	
		# return partial_trace_einsum(x, subsystems, dims)

	
def xOtimesI(A, subsystems, fulldims, checks = False):
	if set(subsystems) == {s+1 for s in range(len(fulldims))}:
		assert np.prod(A.shape)==1, f"tensor with id for all subsystems is defined for scalars only, A.shape = {A.shape}"
		A.shape = (1,1)
		return A * np.identity(np.prod(fulldims))
	elif subsystems == []:
		return A
	else:
		# return xOtimesI_bc_multi(A, subsystems, fulldims, checks = False)
		return xOtimesI_bc_multi_no_inds(A, fulldims, *xOtimesI_bc_multi_inds(subsystems, fulldims))
	



	
def partial_trace_einsum(x, subsystems, dims, checks = False):
	
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



def partial_trace_inds(subsystems, dims):
	'''
	compute inds for _einsum version
	when the same map is applied many times...
	'''
	kept_sys = [ (i not in subsystems) for i in range(1,len(dims)+1)]
	
	dim_out = np.prod(np.array(dims)[kept_sys])
		
	# inds for np.einsum
	inds_in = np.arange(1,2*len(dims)+1).reshape((2,len(dims)))  # dims = [2,3,2] ---> inds_in= [1,2,3; 4,5,6]
	subsys_inds = np.array(subsystems)-1
	inds_in[1,subsys_inds] = inds_in[0,subsys_inds] # subsystems = [2,3] ---> inds_in = [1,2,3; 4,2,3]
	inds_out  = inds_in[:,np.array(kept_sys)] # [1,2,3; 4,2,3] --> [1;4] i.e. 'ijkljk-->il'
	
	return inds_in.ravel().tolist(), inds_out.ravel().tolist(), dim_out
		


def partial_trace_no_inds(x, dims, inds_in, inds_out, dim_out):
	
	return np.reshape(
				np.einsum( x.reshape(dims + dims), inds_in, inds_out), (dim_out,)*2 
			)


def partial_trace_bc(x, subsystem, dims):
	
	r = np.arange(dims[subsystem])
	inds = tuple(slice(d) if i != subsystem else r for i,d in enumerate(dims))*2
	dims_out = [d for i,d in enumerate(dims) if i!=subsystem ]
	x.shape = tuple(dims)*2
	x = x[inds]
	x = np.sum(x, axis=0)
	return x.reshape((np.prod(dims_out),)*2)




def test_partial_trace(partial_trace_func):
	'''
	test partial_trace funcs
	'''
	print(f"testing partial trace function: '{partial_trace_func.__name__}'", end  ='... ')
	X = np.array([[0,1],[1,0]]) 
	A = np.array([[1,0],[0,3]])
	I = np.identity(4)
	AX = np.kron(A,X)
	AXX= np.kron(AX,X)
	XXA = np.kron(X,np.kron(X,A))

	B = np.arange(9).reshape((3,3))
	try:
		assert np.allclose( partial_trace_func(tensorProd(B,A,I), [3], [3,2,4]), np.trace(I) * tensorProd(B,A))
		assert np.allclose( partial_trace_func(tensorProd(B,A,I), [1], [3,2,4]), np.trace(B) * tensorProd(A,I))
		assert np.allclose( partial_trace_func(tensorProd(B,A,I), [2], [3,2,4]), np.trace(A) * tensorProd(B,I))
		
		assert np.allclose( partial_trace_func(tensorProd(B,A,I), [1,2,3], [3,2,4]), np.trace(A) *np.trace(B) *np.trace(I))
		
		assert np.allclose( partial_trace_func(tensorProd(B,A,I,B), [3,1], [3,2,4,3]), np.trace(I) * np.trace(B) * tensorProd(A,B))
		assert np.allclose( partial_trace_func(tensorProd(B,A,I,B), [1,2], [3,2,4,3]), np.trace(B) * np.trace(A) * tensorProd(I,B))
		assert np.allclose( partial_trace_func(tensorProd(B,A,I,B), [2,3], [3,2,4,3]), np.trace(A) * np.trace(I) * tensorProd(B,B))
	except AssertionError:
		raise
	else:
		print('passed')

def xOtimesI_LR(left_dim, x, right_dim ):
	if right_dim > 1:
		tmp = xOtimesI_4D_AxI(x, right_dim, x.shape[0])
		if left_dim > 1:
			return xOtimesI_4D_IxA(tmp,left_dim, tmp.shape[0])
		else:
			return tmp
	elif left_dim > 1:
		return xOtimesI_4D_IxA(x,left_dim, x.shape[0])
	elif (left_dim,right_dim) == (1,1):
		return x 
	# right first is faster
	
	
	
def xOtimesI_4D_AxI(A, d, m):
	out = np.zeros((m,d,m,d),dtype=A.dtype)
	r = np.arange(d)
	out[:,r,:,r] = A
	out.shape = (m*d,m*d)
	return out
    
def xOtimesI_4D_IxA(A, d, m):
	out = np.zeros((d,m,d,m),dtype=A.dtype)
	r = np.arange(d)
	out[r,:,r,:] = A
	out.shape = (m*d,m*d)
	return out


def xOtimesI_kron_perm(x, subsystems, fulldims, checks = False):
	''' 
	adjoint of partial trace: adds \otimes \id terms in specified subsystems
	subsystems are counted from 1 
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
	
	
	
def xOtimesI_bc_perm(x, subsystems, fulldims, checks = False):
	''' 
	adjoint of partial trace: adds \otimes \id terms in specified subsystems
	subsystems are counted from 1 
	'''
	kept_sys = [ (i not in subsystems) for i in range(1,len(fulldims)+1)]
	
	kept_dims = [fulldims[i] for i in range(len(fulldims)) if kept_sys[i]]
	traced_dims = [fulldims[i] for i in range(len(fulldims)) if not kept_sys[i]]
	 
	if checks:
		assert x.shape[0] == x.shape[1] and x.shape[0] == np.prod(np.array(fulldims)[kept_sys]),\
			f"x has wrong dimensions: x.shape = {x.shape}, dimsfull= {dimsfull}, subsystems= {subsystems}"
	
	# id terms are added on the right and then permuted to their place 
	
	Ix = xOtimesI_4D_IxA(x, np.prod(traced_dims), np.prod(kept_dims)).reshape( traced_dims+ kept_dims +  traced_dims + kept_dims  )
		
	traced_sys = [ not b for b in kept_sys]
	
	perm_l = np.zeros((len(fulldims)),dtype = int)
	perm_l[traced_sys] = range(sum(traced_sys))
	perm_l[ kept_sys] = range(sum(traced_sys),len(fulldims))
	perm_r = np.zeros((len(fulldims)),dtype = int)
	perm_r[traced_sys] = [i+len(fulldims) for i in range(sum(traced_sys))]
	perm_r[ kept_sys] = [i+len(fulldims) for i in range(sum(traced_sys),len(fulldims))]
	
	perm  = perm_l.tolist() + perm_r.tolist()
	
	if checks: 
		assert fulldims + fulldims == [(kept_dims + traced_dims + kept_dims + traced_dims)[i] for i in perm ], 'permutation check!!'
	
	return np.transpose(Ix, axes = perm).reshape( (np.prod(fulldims),np.prod(fulldims)) )



def xOtimesI_bc(A, subsystems, fulldims, checks = False):
	assert len(subsystems) == 1 , "only one subsystem supported"
	pos = subsystems[0]-1 # indexing from 0 in this function
	d_I = fulldims[pos]
	dimsA = [d for i,d in enumerate(fulldims) if i != pos]
	 
	
	if checks:
		assert A.shape[0] == np.prod(dimsA), 'dims should multiply to the dimensions of A'
		assert  0 <= pos <= len(dimsA), f'position can be between 0 and len(dims). pos = {pos}, dims = {dims}'

	dims = dimsA[:]
	dims.insert(pos,d_I)
	out = np.zeros(tuple(dims + dims))
	r = np.arange(d_I)
	out[tuple(slice(d) if i != pos else r for i,d in enumerate(dims)) * 2] = A.reshape(tuple(dimsA) * 2)
	out.shape = (np.prod(dims),np.prod(dims))
	return out


def xOtimesI_bc_multi(A, subsystems, fulldims, checks = False):
	
	
	pos = [s-1 for s in subsystems] # indexing from 0 in this function
	dimsA = [d for i,d in enumerate(fulldims) if not (i in pos)]
	dimsId = [d for i,d in enumerate(fulldims) if (i in pos)]
	
	# print(f"dimsA = {dimsA}")
	# print(f"dimsId = {dimsId}")
	
	if checks:
		assert A.shape[0] == np.prod(dimsA), 'dims should multiply to the dimensions of A'
	
	
	out = np.zeros(tuple(fulldims + fulldims))
	rs = np.unravel_index(list(range(np.prod(dimsId))), tuple(dimsId))
	inds = []
	j=0
	for i,d in enumerate(fulldims):
		if i in pos:
			inds.append(rs[j]) 
			j+=1
		else:
			inds.append(slice(d))
		
	out[tuple(inds)*2] = A.reshape(tuple(dimsA) * 2)
	out.shape = (np.prod(fulldims),np.prod(fulldims))
	return out


def xOtimesI_bc_multi_inds(subsystems, fulldims):
	
	pos = [s-1 for s in subsystems] # indexing from 0 in this function
	dimsA = [d for i,d in enumerate(fulldims) if not (i in pos)]
	dimsId = [d for i,d in enumerate(fulldims) if (i in pos)]
	
	rs = np.unravel_index(list(range(np.prod(dimsId))), tuple(dimsId))
	inds = []
	j=0
	for i,d in enumerate(fulldims):
		if i in pos:
			inds.append(rs[j]) 
			j+=1
		else:
			inds.append(slice(d))
		
	return inds, dimsA

def xOtimesI_bc_multi_no_inds(A, fulldims, inds, dimsA, checks = False):
	
	if checks:
		assert A.shape[0] == np.prod(dimsA), 'dims should multiply to the dimensions of A'
	out = np.zeros((np.prod(fulldims),)*2)
	out.shape = tuple(fulldims + fulldims)
	out[tuple(inds)*2] = A.reshape(tuple(dimsA) * 2)
	out.shape = (np.prod(fulldims),np.prod(fulldims))
	# np.reshape(out, (np.prod(fulldims),np.prod(fulldims)))
	return out


def test_xOtimesI(xOtimes_func,N):	
	print(f"testing xOtimes function: '{xOtimes_func.__name__}' ", end = '... ')
	try:
		d1 =4
		dimsA = [d1,2]
		A1 = np.random.rand(d1,d1)
		X = np.array([[0,1],[1,0]])
		A = np.kron(A1, X)

		d_I = 4
		IxA1xX = tensorProd(np.identity(d_I), A1,X)
		pos = 0
		B0 = xOtimes_func(A, [1], [d_I,d1,2])
		assert np.allclose(IxA1xX,B0)
		
		assert np.allclose(A, xOtimes_func(A, [3], [d1,2,1]))
		assert np.allclose(A, xOtimes_func(A, [1], [1,d1,2]))
		assert np.allclose(np.identity(d_I), xOtimes_func(np.array([[1]]), [2], [1,d_I]))
		assert np.allclose(np.identity(d_I), xOtimes_func(np.array([[1]]), [1], [d_I,1]))


		A1xIxX = tensorProd(A1, np.identity(d_I), X)
		pos = 1
		B1 = xOtimes_func(A,[2], [d1,d_I,2])
		assert np.allclose(A1xIxX,B1)

		A1xXxI = tensorProd(A1, X, np.identity(d_I)) 
		pos = 2
		B2 = xOtimes_func(A,[3], [d1,2,d_I])
		assert np.allclose(A1xXxI,B2)
		
		X = np.array([[0,1],[1,0]])
		A = np.arange(16).reshape((4,4))
		I2 = np.identity(2)
		I3 = np.identity(3)
		I4 = np.identity(4)
		
		
		
		assert np.allclose( xOtimes_func(tensorProd(A,X),[2],[4,2,2]) , tensorProd(A,I2,X))
		assert np.allclose( xOtimes_func(tensorProd(A,X),[3],[4,2,3]) , tensorProd(A,X,I3))
		assert np.allclose( xOtimes_func(tensorProd(A,X),[1],[3,4,2]) , tensorProd(I3,A,X))
		
		assert np.allclose( xOtimes_func(tensorProd(A,X,A),[1], [3,4,2,4]) , tensorProd(I3,A,X,A))
		assert np.allclose( xOtimes_func(tensorProd(A,X,A),[2], [4,3,2,4]) , tensorProd(A,I3,X,A))
		assert np.allclose( xOtimes_func(tensorProd(A,X,A),[3], [4,2,3,4]) , tensorProd(A,X,I3,A))
		assert np.allclose( xOtimes_func(tensorProd(A,X,A),[4], [4,2,4,3]) , tensorProd(A,X,A,I3))
		
		X = np.array([[0,1],[1,0]])
		A = np.arange(16).reshape((4,4))
		I2 = np.identity(2)
		I3 = np.identity(3)
		I4 = np.identity(4)
		
		assert np.allclose( xOtimes_func(tensorProd(A,X),[2,3],[4,2,2,2]) , tensorProd(A,I2,I2,X))
		
		assert np.allclose( xOtimes_func(tensorProd(A,X,A),[1,3,6], [3,4,2,2,4,3]) , tensorProd(I3,A,I2,X,A,I3))
		
		assert np.allclose( xOtimes_func(tensorProd(A,X,A),[1,2,5,7], [3,2,4,2,2,4,3]) , tensorProd(I3,I2,A,X,I2,A,I3))
		
		# dims = [2,33,2]
		# for rep in range(N):
			# for subset in [[1],[len(dims)]]:
				# subset = [1]
				# dimsA = [d for i,d in enumerate(dims) if not i+1 in subset ]
				# A = np.random.rand( np.prod(dimsA),np.prod(dimsA))
								
				# AxI = xOtimes_func(A, subset, dims)
		for rep in range(N):			
			dims = [2,2,2,2,2]	
			stuff = list(range(1,len(dims)+1))
			M = np.random.rand(np.prod(dims),np.prod(dims))
			for L in range(len(stuff) + 1):
				for subset in itertools.combinations(stuff, L):
					# print(list(subset))
					ptM = partial_trace(M, list(subset), dims)
					dimsA = [d  for j,d in enumerate(dims) if not j+1 in subset] if list(subset) != list(range(1,len(stuff)+1)) else 1
					A = np.random.rand( np.prod(dimsA),np.prod(dimsA))
										
					AxI = xOtimes_func(A, list(subset), dims)
						
					assert  np.allclose(np.trace(ptM @ A),np.trace(M @ AxI))
		
		
	except:
		raise
	else:
		print('passed')

def bench_xOtimes_LR(dims, N, xOtimes_func=xOtimesI):

	for rep in range(N):
		for subset in [[1],[len(dims)]]:
			dimsA = [d for i,d in enumerate(dims) if not i+1 in subset ]
			A = np.random.rand( np.prod(dimsA),np.prod(dimsA))
			if subset == [1]:
				AxI = xOtimesI_LR(dims[0], A, 1 )
			elif subset == [len(dims)]:
				AxI = xOtimesI_LR(1, A, dims[-1] )
				
			AxI2 = xOtimes_func(A, subset, dims)
			assert np.allclose(AxI,AxI2)
					

def xOtimesI_inds(subsystems, fulldims):
	''' 
	computes indices for adjoint of partial trace
	'''
	kept_sys = [ (i not in subsystems) for i in range(1,len(fulldims)+1)]
	
	kept_dims = [fulldims[i] for i in range(len(fulldims)) if kept_sys[i]]
	traced_dims = [fulldims[i] for i in range(len(fulldims)) if not kept_sys[i]]
	 
	 	
	dim_I_in_xI = np.prod(	[ d for i,d in enumerate(fulldims) if not kept_sys[i]]	)
	shape_for_reshape_xI = kept_dims + traced_dims + kept_dims + traced_dims 
		
	
	perm_l = np.zeros((len(fulldims)),dtype = int)
	perm_l[kept_sys] = range(sum(kept_sys))
	perm_l[[ not b for b in kept_sys] ] = range(sum(kept_sys),len(fulldims))
	perm_r = np.zeros((len(fulldims)),dtype = int)
	perm_r[kept_sys] = [ i+ len(fulldims) for i in range(sum(kept_sys))]
	perm_r[[ not b for b in kept_sys] ] = [ i+ len(fulldims) for i in range(sum(kept_sys),len(fulldims)) ]
	
	axes_for_transpose  = perm_l.tolist() + perm_r.tolist()
	
	return dim_I_in_xI, shape_for_reshape_xI, axes_for_transpose



def xOtimesI_no_Inds(x, dim_I_in_xI, shape_for_reshape_xI, axes_for_transpose, totaldim):
	''' 
	adjoint of partial trace: adds \otimes \id terms in specified subsystems
	'''
	
	# id terms are added on the right and then permuted to their place 
	xI = np.kron(x,\
			np.identity(dim_I_in_xI	)
		).reshape( shape_for_reshape_xI )
		
	return np.transpose(xI, axes = axes_for_transpose).reshape( (totaldim,)*2 )
	




def apply_multiple_kraus_kron(x, IKI):
	dim_out = IKI[0].shape[0]
		
	out = np.zeros((dim_out,dim_out), dtype = x.dtype)
	for iki in IKI:
		out += iki @ x @ iki.conj().T
		
	return out
	
def apply_single_kraus_kron(x, IKI):
	return IKI[0] @ x @ IKI[0].conj().T
		


def  apply_kraus(x, dims, kraus, subsystem, checks = False):
	'''
	apply map to single subsystem of x: 
	x --> sum_i k_i x k_i^\dagger
	* ((subsystem is counted from 0!!!!!!!!!!))
	for now the function supports only consecutive blocks of spins 
	i.e. inds_to_act_on = [0,1,0,1] is not allowed 
	'''
	assert len(dims)<25, 'x has to many axes. (np.einsum supports up to 52)'
	
	
	dim_out = int( np.prod([d for i,d in enumerate(dims) if i != subsystem ]) * kraus[0].shape[0])
	
	
	y = np.zeros((dim_out,dim_out), dtype = x.dtype)
	
	 
	inds_r = list(range(len(dims)*2))
	inds_r[subsystem] = 50
	inds_l = inds_r.copy()
	inds_r[subsystem + len(dims)] = 51
	
	for K in kraus:
		
		# template for what is coded below:
		# y1 = np.einsum(\
				# K,[1,11],\
				# np.einsum(\
					# x.reshape((dims + dims)), [0,11,2, 3,44,5],\
					# K.conj().T, [44,4],\
					# [0,11,2,3,4,5]
				# ), [0,11,2,3,4,5],\
				# [0,1,2,3,4,5]\
			# )
		# print(y1.shape)
		# y += y1.reshape((dim_out,dim_out))
		
		y += np.einsum(\
				K,[subsystem,50],\
				np.einsum(\
					x.reshape((dims + dims)), inds_r,\
					K.conj().T, [51,subsystem + len(dims)],\
					inds_l
				), inds_l,\
				list(range(len(dims)*2))\
			).reshape((dim_out,dim_out))
		
	return y
	


def test_apply_kraus():

	'''
	test apply_kraus
	'''
 
	 
	 
	dims = [4,4,4,4,4]
	A = np.random.rand( np.prod(dims), np.prod(dims))
	kraus = [np.random.rand(3,4**3), ]
	
	I4 = np.identity(4)
	I4s = sps.identity(4)
	
	for j in range(100): 
	 

		expl_prod = np.zeros((4*3*4, 4*3*4))
		for K in kraus:
			IKI = tensorProd(I4,K,I4)
			expl_prod += IKI @ A @ IKI.conj().T
		
		expl_prod_s = sps.csr_matrix((4*3*4, 4*3*4) )
		for K in kraus:
			IKI = sps.kron(sps.kron(I4s,K), I4s)
			expl_prod_s += IKI @ A @ IKI.conj().T
		
		using_apply = apply_kraus(A, [4, 4*4*4, 4], kraus, 1)
		
		print(np.allclose( expl_prod, using_apply ))
		print(np.allclose( expl_prod_s, using_apply ))

			
			

	# dims = [2,4,3,5]
	# A = np.random.rand( np.prod(dims), np.prod(dims))
	# kraus = [np.random.rand(3,2),np.random.rand(3,2),np.random.rand(3,2),np.random.rand(3,2) ]
	# I2 = np.identity(2)
	# I5 = np.identity(5)
	# I4 = np.identity(4)
	# I3 = np.identity(3)

	# expl_prod = 0
	# for K in kraus:
		# KIII = tensorProd(K,I4,I3,I5)
		# expl_prod += KIII @ A @ KIII.conj().T

	# print(np.allclose( expl_prod, apply_kraus(A, dims, kraus, 0)))

# test_apply_kraus()
	
# @profile
def apply_cg_maps(x, dims_x, kraus, action_pattern_in, checks = False):
	'''
	apply x --> sum_i K_i x K_i^\dagger for K_i in karus according to action_pattern.
	action_pattern specifies on which indices of x each copy of the map acts, e.g. :
	x is a 4-body state and the map acts on inds [1,2] of [0,1,2,3] then 
		action_pattern = [0,1,1,0].
	x is a 4-body state and the *2 copies* of the map act on [0,1] and [2,3] then 
		action_pattern = [1,1,2,2].
	
	for now the function supports only consecutive blocks of spins 
		(i.e. *not* action_pattern = [0,1,0,1]),
	and only one map.
	'''
	action_pattern = list(action_pattern_in)

	n_copies = max(action_pattern)
	
	map_in_dim = kraus[0].shape[1]
	map_out_dim = kraus[0].shape[0]
	
	
	dims = list(dims_x)
	out = x
	
	# print(f'action_pattern = {action_pattern}')
	# print(f'dims = {dims}')
	
	cs = [c for c in range(1,n_copies+1) if c in action_pattern]
	
	
	for c in cs:
		# loop over action_pattern applying one map at a time
		
		# print(f'c = {c}')
		
		dims_to_act_on = [ dims[i] for i,a in enumerate(action_pattern) if a == c]
		
		# print(f'dims_to_act_on = {dims_to_act_on}')
		
		if checks:
			assert np.prod(dims_to_act_on) == map_in_dim, f"in applying copy {c} of the map: action patter and dims dont match.\n" +\
				f"ind to act on has dims={np.prod(dims_to_act_on)} and dims_map={[kraus[0].shape]}"
	
		
		# update action pattern and dims lists, eg dims = [d,d,d,d] action pattern = [1,1,2,2]
		first_pos = next(i for i in range(len(action_pattern)) if action_pattern[i] == c)
		
		# print(f'first_pos = {first_pos}')
		
		action_pattern[first_pos] = 0  # action_pattern = [0,1,2,2]
		# print(f'dims = {dims}')
		dims[first_pos] = map_out_dim  # dims =			  [D,d,d,d]
		dims = [ d for i,d in enumerate(dims) if action_pattern[i] != c ] # dims = [D,d,d]
		
		# print(f'dims = {dims}')
		
		dims_joint = dims.copy()
		dims_joint[first_pos] = np.prod(dims_to_act_on) # dims_joint = [d^2,d,d]
		
		# print(f'dims_joint = {dims_joint}')
		
		action_pattern = [a for a in action_pattern if a != c] # action pattern = [0,2,2]
		
		# print(f'action_pattern = {action_pattern}')
		
		# print('-----------------------------------------------')
		out = apply_kraus(out, dims_joint, kraus, subsystem = first_pos )   #subsystem is counted from 0!!!!!!!
		
	return out
		
		
		
	
	
'''
test apply_cg_maps
'''

		
# dims = [2,2,2,2]
# A = np.random.rand( np.prod(dims), np.prod(dims))
# action_pattern = [0,0,1,1]
# kraus = [np.random.rand(3,4), ]	

# direct_calc = apply_kraus(A, [2,2,4], kraus, 2)
# # print(direct_calc)
# test_calc = apply_cg_maps(A, dims, kraus, action_pattern, checks= True)
# # print(test_calc)
# print(np.allclose( direct_calc,test_calc))
	

		
# dims = [2,2,2,2]
# A = np.random.rand( np.prod(dims), np.prod(dims))
# action_pattern = [3,3,1,1]
# kraus = [np.random.rand(3,4), ]	

# direct_calc = apply_kraus(\
	# apply_kraus(A, [2,2,4], kraus, 2),\
	# [4,3], kraus,0) 

# # print(direct_calc)
# test_calc = apply_cg_maps(A, dims, kraus, action_pattern, checks= True)
# # print(test_calc)
# print(np.allclose( direct_calc,test_calc))
	


		
# dims = [3,2,2,3]
# A = np.random.rand( np.prod(dims), np.prod(dims))
# action_pattern = [1,0,0,2]
# kraus = [np.random.rand(3,6), np.random.rand(3,6) ]	

# direct_calc = apply_kraus(\
	# apply_kraus(A, [3,2,2,3], [k.T for k in kraus], 0),\
	# [6,4,3], [k.T for k in kraus], 2) 

# # print(direct_calc)
# test_calc = apply_cg_maps(A, dims, [k.T for k in kraus], action_pattern, checks= True)
# # print(test_calc)
# print(np.allclose( direct_calc,test_calc))


	

		
# dims = [2,3,2,2,3,2]
# A = np.random.rand( np.prod(dims), np.prod(dims))
# action_pattern = [1,1,0,0,2,2]
# kraus = [np.random.rand(3,6), ]	

# direct_calc = apply_kraus(\
	# apply_kraus(A, [6,2,2,6], kraus, 0),\
	# [3,4,6], kraus,2) 

# # print(direct_calc)
# test_calc = apply_cg_maps(A, dims, kraus, action_pattern, checks= True)
# # print(test_calc)
# print(np.allclose( direct_calc,test_calc))


		
# dims = [2,3,2,2,3,2]
# A = np.random.rand( np.prod(dims), np.prod(dims))
# action_pattern = [0,0,0,0,0,0]
# kraus = [np.random.rand(3,6), ]	

# # direct_calc = apply_kraus(\
	# # apply_kraus(A, [6,2,2,6], kraus, 0),\
	# # [3,4,6], kraus,2) 

# # print(direct_calc)
# test_calc = apply_cg_maps(A, dims, kraus, action_pattern, checks= True)
# # print(test_calc)
# print(np.allclose( A,test_calc))
	
		
	


	
if __name__ == '__main__':
	main()
