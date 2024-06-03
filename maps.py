
import numpy as np
import copy

import matrix_aux_functions as mf
   
 
class Maps:
	"""
	
	"""
	check_inputs = True
	
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
		
		
	def __call__(self, in_var,  in_vec):
		""" 
		applies the map to the components of in_vec[in_var.indices...] and returns the result
		"""
		
		if in_var.complex:
			matrix_var = mf.vec2mat(\
				dim = np.prod(in_var.dims),\
				real_part = in_vec[in_var.indices_real[0] : in_var.indices_real[1]+1 ],\
				imag_part = in_vec[in_var.indices_imag[0] : in_var.indices_imag[1]+1 ],\
				check_inputs = Maps.check_inputs
			)
		else:
			matrix_var = mf.vec2mat(\
				dim = np.prod(in_var.dims),\
				real_part = in_vec[in_var.indices_real[0] : in_var.indices_real[1]+1 ],\
				check_inputs = Maps.check_inputs
			)
		
		if self.adjoint_flag:
			return self.apply_adj(matrix_var)
		else:
			return self.apply(matrix_var)
		
		
		
		
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
		):
				
		dims = {'in': np.prod(action['dims_in']), 'out': np.prod(action['dims_out'])} # total dimensions
		super().__init__(name, dims, adj_name = name + '^*')
		self.kraus = kraus
		self.action = action
		
	def apply(self, x, checks = False):
		return mf.apply_cg_maps(x, self.action['dims_in'], self.kraus, self.action['pattern'], checks)
	
	def apply_adj(self, x, checks = False):
		return mf.apply_cg_maps(x, self.action['dims_out'], [k.conj().T for k in self.kraus] , self.action['pattern_adj'], checks)
	
	
	
	
	
	
	
			
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
		
		return np.trace(self.operator.T @ x)
	
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
		state_dims #  dims of input
		):
				
		name = f"Trace_[{subsystems}/{len(state_dims)}]"
		remaining_subsystems = {i+1 for i in range(len(state_dims))}.difference(subsystems)
		dims = {'in': np.prod(state_dims),
				'out': np.prod( [state_dims[i-1] for i in remaining_subsystems] ) }
		super().__init__(name, dims, adj_name = f'(x)Id_[{subsystems}/{len(state_dims)}]')
		self.state_dims = state_dims
		self.subsystems = subsystems
		
	

	def apply(self, x):
		return mf.partial_trace(x, self.subsystems, self.state_dims)
		
	def apply_adj(self, x):
		return mf.xOtimesI(x, self.subsystems, self.state_dims)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	
	
	
