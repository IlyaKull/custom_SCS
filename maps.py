
import numpy as np
import copy

import matrix_aux_functions as mf
   
 
class Maps:
	"""
	
	"""
	check_inputs = False
	verbose= False
	
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
				
		return s
		
 
	def __call__(self, var, v_in, sign , out ):
		""" 
		apply map on v (or adjoint of map if adjoint_flag == True)
		ALWAYS ACT IN PLACE (+=)
		"""
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
	!!!!!!!!!! TODO:::::::::
	move index calculations from matrix_aux_funcs module into self 
	(so that it computes once at init and not at every call)
	"""
	def __init__(self,  
		subsystems,  # subsystems to trace : set
		state_dims #  dims of input
		):
				
		name = f"Trace_[{subsystems}/{len(state_dims)}]"
		remaining_subsystems = {i+1 for i in range(len(state_dims))}.difference(subsystems)
		dims = {'in': np.prod(state_dims),
				'out': np.prod( [state_dims[i-1] for i in remaining_subsystems] ) }
		super().__init__(name, dims, adj_name = f'(x)Id_[{subsystems}/{len(state_dims)}]')
		self.state_dims = state_dims
		self.subsystems = subsystems
		
		# pre compute inds for pTr op
		self.pTr_inds_in, self.pTr_inds_out, self.pTr_dim_out = mf.partial_trace_inds(subsystems, state_dims)
		
		
		# pre compute inds for adjoint op
		self.xI_dim_I, self.xI_shape_for_reshape, self.xI_axes_for_transpose = \
			mf.xOtimesI_inds(subsystems, state_dims)
		
		self.totaldim = np.prod(state_dims)
		
		
	def apply(self, x):
		
		return mf.partial_trace_no_inds(x, self.state_dims, self.pTr_inds_in, self.pTr_inds_out, self.pTr_dim_out)
		
		
	def apply_adj(self, x):
		
		return mf.xOtimesI_no_Inds(x, self.xI_dim_I, self.xI_shape_for_reshape, self.xI_axes_for_transpose, self.totaldim)
	
	
	
 
		
		
		
		
		
		
		
		
		
		
		
	
	
	
