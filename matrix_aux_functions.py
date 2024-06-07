
		
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
	


def vec2mat(dim, vec):
	'''
	convention is to store the upper triangular part in row major form 
	(= to lower triangular part in column major form)
	'''
	assert len(vec) == dim_symm_matrix(dim), \
		f'vec has wrong number of entries: len(vec) = {len(vec)}, dimSymmMat = {dim_symm_matrix(dim)}'
	
	
	mat = np.zeros((dim,dim), dtype = vec.dtype )
	
	mat[np.triu_indices(dim)] += vec
	mat += mat.T.conj()
	mat[np.diag_indices(dim)] /= 2.0
	
				
	return mat




def mat2vec(dim, mat):
	assert mat.shape == (dim, dim), 'check simensions of input matrix'
	
	return mat[np.triu_indices(dim)]
	
	 


 
# ################ Testing vec2mat

# vec = np.arange(10.)
 
# print(f'vec = \n{vec}') 

# print(f'mat = \n{vec2mat(4, vec)}')

# print(f'vecFmat = \n{mat2vec(4,vec2mat(4, vec))}')
 





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
 


	
	
	
	
	
	
def  apply_kraus(x, dims, kraus, subsystem, checks = False):
	'''
	apply map to single subsystem of x: 
	x --> sum_i k_i x k_i^\dagger
	* ((subsystem is counted from 0!!!!!!!!!!))
	for now the function supports only consecutive blocks of spins 
	i.e. inds_to_act_on = [0,1,0,1] is not allowed 
	'''
	assert len(dims)<25, 'x has to many axes. (np.einsum supports up to 52)'
	
	dim_out = np.prod([d for i,d in enumerate(dims) if i != subsystem ]) * kraus[0].shape[0]
	
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
	
	
'''
>>> a
[1, 2, 1, 1]
>>> b = [ range(10)[i] if el!=2 else 100 for i,el in enumerate(a)]
>>> b
[0, 100, 2, 3]
'''


# # '''
# # test apply_kraus
# # '''

# # dims = [2,4,2]
# # A = np.random.rand( np.prod(dims), np.prod(dims))
# # kraus = [np.random.rand(4,4), ]
# # I2 = np.identity(2)

# # expl_prod = 0
# # for K in kraus:
	# # IKI = tensorProd(I2,K,I2)
	# # expl_prod += IKI @ A @ IKI.conj().T

# # print(np.allclose( expl_prod, apply_kraus(A, dims, kraus, 1)))

 
 
# # dims = [2,4,3,5]
# # A = np.random.rand( np.prod(dims), np.prod(dims))
# # kraus = [np.random.rand(2,5), ]
# # I2 = np.identity(2)
# # I4 = np.identity(4)
# # I3 = np.identity(3)

# # expl_prod = 0
# # for K in kraus:
	# # IIIK = tensorProd(I2,I4,I3,K)
	# # expl_prod += IIIK @ A @ IIIK.conj().T

# # print(np.allclose( expl_prod, apply_kraus(A, dims, kraus, 3)))

		
		

# # dims = [2,4,3,5]
# # A = np.random.rand( np.prod(dims), np.prod(dims))
# # kraus = [np.random.rand(3,2),np.random.rand(3,2),np.random.rand(3,2),np.random.rand(3,2) ]
# # I2 = np.identity(2)
# # I5 = np.identity(5)
# # I4 = np.identity(4)
# # I3 = np.identity(3)

# # expl_prod = 0
# # for K in kraus:
	# # KIII = tensorProd(K,I4,I3,I5)
	# # expl_prod += KIII @ A @ KIII.conj().T

# # print(np.allclose( expl_prod, apply_kraus(A, dims, kraus, 0)))

	
	
def apply_cg_maps(x, dims_x, kraus, action_pattern, checks = False):
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
	
	n_copies = max(action_pattern)
	
	map_in_dim = kraus[0].shape[1]
	map_out_dim = kraus[0].shape[0]
	
	
	dims = dims_x.copy()
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
	
		
	

