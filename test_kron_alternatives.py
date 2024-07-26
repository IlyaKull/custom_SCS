import numpy as np
import timeit
import scipy as sp

import sys
import line_profiler



profile = line_profiler.LineProfiler()
profile.enable_by_count()
	
	
def _0_np_kron_IxA(A,d,m):
	return np.kron(np.identity(d), A )

def _0_np_kron_AxI(A,d,m):
	return np.kron(A, np.identity(d) )


# def _0_sp_kron_sparse_csr(A,d):
	# A_sparse = sp.sparse.csr_array(A)
	# return sp.sparse.kron(sp.sparse.identity(d), A_sparse ).toarray()

# def _0_sp_kron_sparse_full(A,d):
	# return sp.sparse.kron(sp.sparse.identity(d), A ).toarray()

def _0_sp_la_block_diag_IxA(A,d,m):
	return sp.linalg.block_diag(*[A]*d)	

# def _0_sp_kron_sparse_coo(A,d):
	# A = sp.sparse.coo_array(A)
	# return sp.sparse.kron(sp.sparse.identity(d), A ).toarray()

# def _0_sp_block_diag_coo(A,d):
	# A = sp.sparse.coo_array(A)
	# return sp.sparse.block_diag((A,)*d ).toarray()

# def _0_sp_block_diag_full(A,d):
	# return sp.sparse.block_diag((A,)*d ).toarray()

def _0_np_block_IxA(A,d,m):
	Z = np.zeros(A.shape)
	return np.block([ [Z,]*(j) + [A,] + [Z,]*(d-1-j) for j in range(d) ])
					

def _0_4D_kron_AxI(A, d, m):
	out = np.zeros((m,d,m,d),dtype=A.dtype)
	r = np.arange(d)
	out[:,r,:,r] = A
	out.shape = (m*d,m*d)
	return out
    
def _0_4D_kron_IxA(A, d, m):
	out = np.zeros((d,m,d,m),dtype=A.dtype)
	r = np.arange(d)
	out[r,:,r,:] = A
	out.shape = (m*d,m*d)
	return out

def _1_IxAxI_left_first(A,d,m):
	tmp = _0_4D_kron_IxA(A,d,m)
	return _0_4D_kron_AxI(tmp,d,d*m)
	
def _1_IxAxI_right_first(A,d,m):
	tmp = _0_4D_kron_AxI(A,d,m)
	return _0_4D_kron_IxA(tmp,d,d*m)





profile.add_function(_0_4D_kron_IxA)
profile.add_function(_0_4D_kron_AxI)
profile.add_function(_1_IxAxI_left_first)
profile.add_function(_1_IxAxI_right_first)

# print(profile.__dir__())

current_module = sys.modules[__name__]
# print(dir(current_module))

control_str = '_1_'

func_names_list = [f for f in current_module.__dir__() if f[0:3] == control_str]
# print(func_names_list)
func_list = [current_module.__dict__[f] for f in func_names_list]
# print(func_list)

m = 2 
d = 2 
A = np.random.rand(m,m)
for f in func_list:
	print(f.__name__[3:].ljust(30), '\n' , f(A,d,m))




for d,m in zip((3,),(2000,)):

	A = np.random.rand(m,m)
	 
	 

	print('='*25, f'   TIME d={d} m={m}  ', '='*25)

	for f in func_list:
		print(f.__name__[3:].ljust(30), timeit.timeit(lambda: f(A,d,m), number = 100))

profile.print_stats()
