
import numpy as np
import copy
import time
import matrix_aux_functions as mf
from util import sign_str

import logging
logger = logging.getLogger(__name__)
 
class Maps:
	"""
	
	"""
	list_all_maps = []
	
	log_calls = False
	
	def __init__(self, 
		name, # the name of the map in your paper notes. Used to for display.
		dims, # input and output dimensions
		adj_name = None, # display name for adjoint operator
		implementation = 'std'
		): 
		self.name = name
		self.dims = dims
		self.adj_name = adj_name

				
		self.implementation = implementation
		
		self.time = 0.
		self.time_adj = 0.
		self.calls = 0
		self.calls_adj = 0
		
		Maps.list_all_maps.append(self)
		logger.debug(f'map {self.name} created')
	
	# def mod_map(self, adjoint = False):
		# '''
		# returns a copy of the  map with the chosen attributes:
			# adjoint = true/false
		# '''
		 
		# s = copy.deepcopy(self)
			# # how to do it nicely with self.__class__( modified self.__dict__)???
			
		
		# if adjoint != self.adjoint_flag:
			# s.adjoint_flag = adjoint
			# s.dims['in'] = self.dims['out']
			# s.dims['out'] = self.dims['in']
			# if self.adj_name:
				# s.name = self.adj_name
				# s.adj_name = self.name
		
		# Maps.list_all_maps.append(s)
				
		# return s
		
 
	def __call__(self, var, v_in, sign, adj_flag , out ):
		""" 
		apply map on v (or adjoint of map if adj_flag == True)
		ALWAYS ACT IN PLACE (+=)
		"""
		if Maps.log_calls:
			t = time.perf_counter()
		
		if logging.root.level <= 5:
			logger.log(5, f' calling map {self.name} on variable {var.name}')
			logger.log(5, f'----------> var dims = {var.dims}')
			
		mat = v_in[var.slice].reshape( (var.matdim, var.matdim) )
			
			
		if  adj_flag:
			out += sign * self.apply_adj( mat ).ravel()
		else:
			out += sign * self.apply( mat ).ravel()
	
		if Maps.log_calls:
			t = time.perf_counter() - t
			self.log_call(t,adj_flag)
			
	def log_call(self, t,adj_flag):
		match adj_flag:
			case False:
				self.time += t
				self.calls += 1
			case True:
				self.time_adj += t
				self.calls_adj += 1
		
		
	@classmethod
	def print_maps_log(cls):
		 
		print('='*100)
		print('MAP-CALLS LOG'.center(100))
		print('-'*100)
		log_columns = lambda m:\
			{"Name" : 	{'val' : m.name, 															'format_str' : '<', 	'group' : '---', 'col_width':30},
				"Name adj":				{'val' : m.adj_name, 										'format_str' : '<', 	'group' : 'adj', 'col_width':30},
				"Impl." : 				{'val' : m.implementation, 									'format_str' : '<', 	'group' : 'all', 'col_width':15},
				"Calls" : 				{'val' : m.calls, 											'format_str' : 'g', 	'group' : '---', 'col_width':15},
				"Calls adj" : 			{'val' : m.calls_adj, 										'format_str' : 'g', 	'group' : 'adj', 'col_width':15},
				"t_total" : 			{'val' : m.time, 											'format_str' : '0.2g', 	'group' : '---', 'col_width':15},
				"t_iter" :				{'val' : m.time / m.calls if m.calls > 0 else 0 ,	 		 'format_str' : '0.2g', 'group' : '---', 'col_width':15},
				"t_total_adj" : 		{'val' : m.time_adj, 										'format_str' : '0.2g', 	'group' : 'adj', 'col_width':15},
				"t_iter_adj" :			{'val' : m.time_adj / m.calls_adj if m.calls_adj > 0 else 0 , 'format_str' : '0.2g', 'group' : 'adj', 'col_width':15},
			}
		
			
		for k,v in log_columns(cls.list_all_maps[0]).items():
			if v['group'] in ['all', '---']:
				print(k.ljust(v['col_width']), end = " | " )
		print('')
				
		for m in cls.list_all_maps:
			for group_str in ['---', 'adj']:
				for v in log_columns(m).values():
					if v['group'] in ['all', group_str]:
						print(format(v['val'], v['format_str']).ljust(v['col_width']), end = " | " )
				print('')
		
		print('='*100)	
				
	
	@classmethod
	def _test_self_adjoint(cls, rng, n_samples = 100, tol = 1e-10):
		"""
		test each of the defined maps for self adjointness
		"""
		maps_max_violation = dict()
		for m in cls.list_all_maps:
			logger.debug(f"map {m.name} testing self-adjiontness")
			map_tests=[]
			
			for n in range(n_samples):
				x = rng.random((m.dims['in'],)*(2 if m.dims['in']>1 else 1))
				x = x/np.linalg.norm(x)
				y = rng.random((m.dims['out'],)*(2 if m.dims['out']>1 else 1))
				y = y/np.linalg.norm(y)
				Mx = m.apply(x)
				M_dag_y = m.apply_adj(y)
				map_tests.append(abs(np.vdot(y, Mx) - np.vdot(M_dag_y, x)) )
				
			maps_max_violation[m.name] = max(map_tests)
						
			try:
				assert maps_max_violation[m.name] < tol , f"map {m.name} failed SA test"
			except AssertionError:
				logger.critical(f"map {m.name} didn't pass SA test: |vdot(y, Mx) - vdot(M_dag_y, x)| = {abs(np.vdot(y, Mx) - np.vdot(M_dag_y, x)) :0.3g}, tol = {tol}")
				raise
			else:
				logger.debug(f"map {m.name} passed self-adjoint test")
		
		return maps_max_violation
					
					
# @profile
class CGmap(Maps):
	"""
	coare-graining map in kraus representation
	"""
	def __init__(self, name, 
		kraus = [],  # the kraus representation of the map: a list of matrices
		action = {'dims_in':[], 'dims_out':[], 'pattern':[],'pattern_adj':[]} # action pattern: 
					# dims_in: list of dimensions of input satate
					# # dims_out: list of dimensions of out satate
					# pattern: 
							# 0 means Id is acting on that system. 
							# integer values for blocks on which copies of the same map act:
							# (0,0,1,1,2,2) means IdxIdxCxC is applied where Id is acting on 1 spin and  C is acting on 2 spins
							# in this case dims_in = [d,d,d,d,d,d] and dims_out = [d,d,D,D]
					# pattern_adj: same as pattern but for adjoint map: in this case (0,0,1,2)
		,
		implementation = 'kron', # which function implements the map: 'contract' using np.einsum, 'kron' by buliding IxKxI
		):
				
		dims = {'in': np.prod(action['dims_in']), 'out': np.prod(action['dims_out'])} # total dimensions
		super().__init__(name, dims, adj_name = name + '^*', implementation = implementation)
		self.kraus = kraus
		
		assert len(action['dims_in']) ==  len(action['pattern']), f"map {self.name} action pattern mismatch dims_in: {action['dims_in']} pattern {action['pattern']}"
		assert len(action['dims_out']) ==  len(action['pattern_adj']), f"map {self.name} action adjoint pattern mismatch dims_out: {action['dims_out']} pattern {action['pattern_adj']}"
		self.action = action
		# self.implementation = implementation
		
		if self.implementation == 'kron':
			assert max(self.action['pattern']) == 1, f"action pattern for map '{self.name}' is incompatible with 'kron' implementation"
			# TODO: calc tensor product of multiple kraus ops: I x K_i x K_j x I			
			first_pos = next(i for i in range(len(self.action['pattern'])) if self.action['pattern'][i] == 1)
			

			self.IxKxI = []
			self.IxKxI_dag = []
			for Kop in kraus:
				terms_to_tensor = []
				for i,a in enumerate(self.action['pattern']):
					if a == 0:
						terms_to_tensor.append(np.identity(self.action['dims_in'][i]))
						
					if a==1 and i==first_pos:
						terms_to_tensor.append(Kop)
						
				
				self.IxKxI.append( mf.tensorProd(terms_to_tensor)	)
				self.IxKxI_dag.append(self.IxKxI[-1].conj().T)	
			
			if len(self.IxKxI) > 1:
				self._apply_impl = self._apply_impl_kraus_multiple
				self._apply_adj_impl = self._apply_adj_impl_kraus_multiple
				
			else:	
				self._apply_impl = self._apply_impl_kraus_single
				self._apply_adj_impl = self._apply_adj_impl_kraus_single
							
		elif self.implementation == 'contract':
			self._apply_impl =  self._apply_impl_contract
			self._apply_adj_impl =  self._apply_adj_impl_contract
			self.kraus_dag = [k.conj().T for k in self.kraus]
			
	def apply(self, x):
		return self._apply_impl(x)
	
	def apply_adj(self, x):
		return self._apply_adj_impl(x)
	
	def _apply_impl_kraus_single(self,x):
		return mf.apply_single_kraus_kron(x, self.IxKxI)
	def _apply_impl_kraus_multiple(self,x):
		return mf.apply_multiple_kraus_kron(x, self.IxKxI)
	def _apply_adj_impl_kraus_single(self,x):
		return mf.apply_single_kraus_kron(x, self.IxKxI_dag)
	def _apply_adj_impl_kraus_multiple(self,x):
		return mf.apply_adj_multiple_kraus_kron(x, self.IxKxI_dag)
	
	def _apply_impl_contract(self,x):
		return mf.apply_cg_maps(x, self.action['dims_in'], 	self.kraus,  self.action['pattern'])
	def _apply_adj_impl_contract(self,x):
		return mf.apply_cg_maps(x, self.action['dims_out'], 	self.kraus_dag , 	self.action['pattern_adj'])
	
			
	 
	
	
	
	
			
class TraceWith(Maps):
	"""
	maps x -> trace(O^\dagger * x) ; its adjoint is then c -> c*O
	"""
	def __init__(self, op_name, 
		operator = np.array([0.0]), # the operator with which the trace is taken
		dim = () # dimension of input state
		):
		 		
		super().__init__(f"{op_name}*", dims = {'in': dim, 'out': 1}, adj_name = op_name)
		self.operator = operator
		self.op_name = op_name
	
	def apply(self, x):
		
		# assert tuple(reversed(self.operator.shape)) == x.shape, f"dimensions of operator {self.op_name} don't match input matrix"
		# assert self.operator.dtype == x.dtype, f"operator {self.op_name} is {self.operator.dtype} and matrix m is {x.dtype}"
	
		return np.trace(self.operator @ x)
	
	def apply_adj(self, x):
		
		# assert x.size == 1, f"to apply the adjoint of trace[{self.op_name} * ] the input should be a scalar, x has size {x.size}"
		# assert self.operator.dtype == x.dtype, f"operator {self.op_name} is {self.operator.dtype} and matrix m is {x.dtype}"
		
		# operator is assumed to be self-adjoint
		return x * self.operator
		
		


class Identity(Maps):
	"""
	maps x -> x 
	"""
	def __init__(self,  dim):
				
		super().__init__('Id', dims = {'in': dim, 'out': dim}, adj_name = 'Id')
		
	def apply(self, x ):
		return	x	
		
	def apply_adj(self, x ):
		return	x	



class Trace(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self,  
		dim = ()  # dimension of input state
		):
		
		super().__init__('Trace', dims = {'in': dim, 'out': 1}, adj_name = '1*')
	
	def apply(self, x):
		return np.trace(x)
	
	def apply_adj(self, x):
		
		# assert x.size == 1, f"to apply the adjoint of trace[*] the input should be a scalar, x has size {x.size}"
		return x * np.identity(self.dims['in'], dtype = float)



class AddMaps(Maps):
	"""
	when several maps need to be applied to the same variable they should be wrapped with this
	"""

	def __init__(self, map_list, sign_list):
		assert  all([map_list[0].dims == m.dims for m in map_list]), f"dimension mismatch when adding maps {[m.name for m in map_list]}"
		name = '{' + ''.join([sign_str(s) + m.name for m,s in zip(map_list, sign_list)]) + '}'
		adj_name = '{' + ''.join([sign_str(s) + m.adj_name for m,s in zip(map_list, sign_list)])  + '}'
		super().__init__(name, map_list[0].dims, adj_name = adj_name, implementation = map_list[0].implementation )
		self.map_list = map_list
		self.sign_list = sign_list
		
	def apply(self, x):
		out = np.zeros( (self.dims['out'],)*2 )
		for m,s in zip(self.map_list, self.sign_list):
			out += s * m.apply(x)
		return out
		
	def apply_adj(self, x):
		out = np.zeros( (self.dims['in'],)*2 )
		for m,s in zip(self.map_list, self.sign_list):
			out += s * m.apply_adj(x)
		return out 
		
		
	
class PartTrace(Maps):
	"""
	maps x -> trace_{subsystems}(x) ;  y -> id_{subsystems} \otimes y
	
	"""
	def __init__(self,  
		subsystems,  # subsystems to trace : set
		state_dims, #  dims of input
		):
				
		name = f"Trace_[{subsystems}/{len(state_dims)}]"
		remaining_subsystems = {i+1 for i in range(len(state_dims))}.difference(subsystems)
		dims = {'in': np.prod(state_dims),
				'out': np.prod( [state_dims[i-1] for i in remaining_subsystems] ) }
		
		try:
			assert subsystems != [] 
			assert set(subsystems) != {s+1 for s in range(len(state_dims))}
			assert all([1 <= s <= len(state_dims) for s in subsystems])
		except AssertionError:
			logger.error(f"map {name}: problem with subsystems = {subsystems}, dims = {state_dims}")
			raise
		
		
		
		super().__init__(name, dims, adj_name = f'(x)Id_[{subsystems}/{len(state_dims)}]' )
		self.state_dims = state_dims
		self.subsystems = subsystems
		
		
		# pre compute inds for pTr op
		self.pTr_inds = mf.partial_trace_inds(subsystems, state_dims)
		# self.totaldim = np.prod(state_dims)
		
		# pre compute inds for xOtimesI op
		self.xOtimes_inds = mf.xOtimesI_bc_multi_inds(subsystems, state_dims)
		
		
	
	def apply(self, x):
		return mf.partial_trace_no_inds(x, self.state_dims, *self.pTr_inds)
		
	def apply_adj(self, x):
		return mf.xOtimesI_bc_multi_no_inds(x.copy(), self.state_dims, *self.xOtimes_inds)
	
	

