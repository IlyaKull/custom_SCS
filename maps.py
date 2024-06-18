
import numpy as np
import copy
import time
import matrix_aux_functions as mf
   
 
class Maps:
	"""
	
	"""
	check_inputs = False
	verbose= False
	list_all_maps = []
	log_calls = True
	
	def __init__(self, 
		name, # the name of the map in your paper notes. Used to for display.
		dims, # input and output dimensions
		adj = False, # adjoint flag: true to apply the adjoint
		adj_name = None, # display name for adjoint operator
		check_inputs = False # class variable to switch on checks and debug
		): 
		self.name = name
		self.dims = dims
		self.adjoint_flag = adj
		self.adj_name = adj_name
		
		if check_inputs:
			Maps.check_inputs = check_inputs
		
		self.time = 0.
		self.time_adj = 0.
		self.calls = 0
		self.calls_adj = 0
		
		Maps.list_all_maps.append(self)
	
	
	def mod_map(self, adjoint = False):
		'''
		returns a copy of the  map with the chosen attributes:
			adjoint = true/false
		'''
		 
		s = copy.deepcopy(self)
			# how to do it nicely with self.__class__( modified self.__dict__)???
			
		
		if adjoint != self.adjoint_flag:
			s.adjoint_flag = adjoint
			s.dims['in'] = self.dims['out']
			s.dims['out'] = self.dims['in']
			if self.adj_name:
				s.name = self.adj_name
				s.adj_name = self.name
		
		Maps.list_all_maps.append(s)
				
		return s
		
 
	def __call__(self, var, v_in, sign , out ):
		""" 
		apply map on v (or adjoint of map if adjoint_flag == True)
		ALWAYS ACT IN PLACE (+=)
		"""
		if Maps.log_calls:
			t = time.perf_counter()
		
		if Maps.verbose:
			if var is None:
				print(f'----------> calling map {self.name} on 1')
			else:
				print(f'----------> calling map {self.name} on variable {var.name}')
				print(f'----------> var dims = {var.dims}')
			
		if var is None:
			mat = np.array(1.0) # this takes care of the H*1 map
		else:
			mat = v_in[var.slice].reshape( (var.matdim, var.matdim) )
			
				
			
		if self.adjoint_flag:
			out += sign * self.apply_adj( mat ).ravel()
		else:
			out += sign * self.apply( mat ).ravel()
	
		if Maps.log_calls:
			t = time.perf_counter() - t
			self.log_call(t)
			
	def log_call(self, t):
		match self.adjoint_flag:
			case False:
				self.time += t
				self.calls += 1
			case True:
				self.time_adj += t
				self.calls_adj += 1
		
		
	
	def print_maps_log(self):
		for m in Maps.list_all_maps:
			print(f"Map {m.name} applied {m.calls} times with t_tot = {m.time:.2g}", end = ' ')
			if m.calls > 0:
				print(f": t/iter = {m.time / m.calls:.2g}")
			else:
				print('')
			
			print(f"...and in adjoint mode: {m.calls_adj} times with t_tot = {m.time_adj:.2g}", end = ' ')
			if m.calls_adj > 0:
				print(f": t/iter = {m.time_adj / m.calls_adj:.2g}")
			else:
				print('')
	
		
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
		**kwargs):
				
		dims = {'in': np.prod(action['dims_in']), 'out': np.prod(action['dims_out'])} # total dimensions
		super().__init__(name, dims, adj_name = name + '^*', **kwargs)
		self.kraus = kraus
		self.action = action
		
	def apply(self, x, checks = False):
		return mf.apply_cg_maps(x, self.action['dims_in'], self.kraus, self.action['pattern'])
	
	def apply_adj(self, x, checks = False):
		# print('applying adjoint CG map')
		# print(self.action['pattern_adj'])
		# print(self.kraus[0])
		# print(self.action['dims_out'])
		return mf.apply_cg_maps(x, self.action['dims_out'], [k.conj().T for k in self.kraus] , self.action['pattern_adj'])
	
	
	
	
	
	
	
			
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
		
		if Maps.check_inputs:
			assert tuple(reversed(self.operator.shape)) == x.shape, f"dimensions of operator {self.op_name} don't match input matrix"
			assert self.operator.dtype == x.dtype, f"operator {self.op_name} is {self.operator.dtype} and matrix m is {x.dtype}"
		
		return np.trace(self.operator.T.conj() @ x)
	
	def apply_adj(self, x):
		
		if Maps.check_inputs:
			assert x.size == 1, f"to apply the adjoint of trace[{self.op_name} * ] the input should be a scalar, x has size {x.size}"
			assert self.operator.dtype == x.dtype, f"operator {self.op_name} is {self.operator.dtype} and matrix m is {x.dtype}"
			
		return x * self.operator
		
		


class Identity(Maps):
	"""
	maps x -> x 
	"""
	def __init__(self,  dim):
				
		super().__init__('Id', dims = {'in': dim, 'out': dim})
		
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
		
		if Maps.check_inputs:
			assert x.size == 1, f"to apply the adjoint of trace[*] the input should be a scalar, x has size {x.size}"
			
		return x * np.identity(self.dims['in'], dtype = float)



class PartTrace(Maps):
	"""
	maps x -> trace_{subsystems}(x) ;  y -> id_{subsystems} \otimes y
	
	"""
	def __init__(self,  
		subsystems,  # subsystems to trace : set
		state_dims, #  dims of input
		implementation = 'xOtimesI' # which function to call for the adjoint operation
		):
				
		name = f"Trace_[{subsystems}/{len(state_dims)}]"
		remaining_subsystems = {i+1 for i in range(len(state_dims))}.difference(subsystems)
		dims = {'in': np.prod(state_dims),
				'out': np.prod( [state_dims[i-1] for i in remaining_subsystems] ) }
		super().__init__(name, dims, adj_name = f'(x)Id_[{subsystems}/{len(state_dims)}]')
		self.state_dims = state_dims
		self.subsystems = subsystems
		self.implementation = implementation 
		
		# pre compute inds for pTr op
		self.pTr_inds_in, self.pTr_inds_out, self.pTr_dim_out = mf.partial_trace_inds(subsystems, state_dims)
		
		self.totaldim = np.prod(state_dims)

		match self.implementation:
			case 'xOtimesI':

				self.xI_dim_I, self.xI_shape_for_reshape, self.xI_axes_for_transpose = \
					mf.xOtimesI_inds(subsystems, state_dims)
				self._adj_impl = lambda x:\
					mf.xOtimesI_no_Inds(x, self.xI_dim_I, self.xI_shape_for_reshape, self.xI_axes_for_transpose, self.totaldim)

			case 'kron':
				try:
					is_left = \
						min(subsystems) == 1
					is_right = \
						max(subsystems) == len(state_dims)
					
					assert is_left or is_right
					
					subsys_contig = \
						set(subsystems) == set(range(min(subsystems), max(subsystems)+1)) 
					
					compl_subsystems = set(range(1,len(state_dims)+1)) - set(subsystems)
					compl_subsys_contig = \
						compl_subsystems == set(range(min(compl_subsystems), max(compl_subsystems)+1)) 
					assert compl_subsys_contig
				except AssertionError:
					print("to use the kron()-based implementation, subsystems must consist of leftmost and/or rightmost contiguous blocks")
					
				else:
					tot_dim_subsys = np.prod([state_dims[i-1] for i in subsystems])
					if subsys_contig:
						if is_left:
							self.id_left = np.identity(tot_dim_subsys)
							self.id_right = []
							self._adj_impl = lambda x: np.kron(self.id_left,x)
							
						if is_right:
							self.id_left = []
							self.id_right = np.identity(tot_dim_subsys)
							self._adj_impl = lambda x: np.kron(x,self.id_right)
					else:
						left_dim = np.prod([state_dims[i-1] for i in range(1,min(compl_subsystems))])
						right_dim = np.prod([state_dims[i-1] for i in range(max(compl_subsystems)+1, len(state_dims) +1 )])
						self.id_left = np.identity(left_dim)
						self.id_right = np.identity(right_dim)
						self._adj_impl = lambda x: np.kron(self.id_left, np.kron(x ,self.id_right))
						

		
		
		
	def apply(self, x):
		return mf.partial_trace_no_inds(x, self.state_dims, self.pTr_inds_in, self.pTr_inds_out, self.pTr_dim_out)
		
		
	def apply_adj(self, x):
		return self._adj_impl(x)
	

