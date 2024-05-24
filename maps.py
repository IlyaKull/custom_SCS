
import numpy as np
import copy
 
class Maps:
	"""
	
	"""
	
	def __init__(self, 
		name, # the name of the map in your paper notes. Used to for display.
		sign, # maps come with a sign (+/-1): determinig the sign with which they appear in a constraint
		dims, # input and output dimensions
		adjoint_flag = False, # adjoint flag: true to apply the adjoint
		adj_name = None, # display name for adjoint operator
		**kwargs): 
		self.name = name
		self.sign = sign
		self.dims = dims
		
		self.adjoint_flag = adjoint_flag
		self.adj_name = adj_name
		
	
	
	def mod_map(self, sign = None, adjoint = False):
		'''
		returns another instance of the  map with modified chosen attributes:
			sign +/- int 
			adjoint = true/false
		'''
		 
		kwargs = self.__dict__
		print(kwargs)
		
		kwargs['adjoint'] = adjoint
		if sign:
			kwargs['sign'] = sign
		
		s = self.__class__(**kwargs)
				
		print(s.__dict__)
		
				
		# print((self.name, self.adjoint_flag, self.sign))
		# print((hex(id(self.name)), hex(id(self.adjoint_flag)), hex(id(self.sign))))
		# print((s.name , s.adjoint_flag, s.sign) )
		# print((hex(id(s.name)), hex(id(s.adjoint_flag)), hex(id(s.sign))))
		'''
		if adjoint != self.adjoint_flag:
			s.adjoint_flag = adjoint
			s.dims['in'] = self.dims['out']
			s.dims['out'] = self.dims['in']
			if self.adj_name:
				s.name = self.adj_name
				s.adj_name = self.name
		
		if sign:
			s.sign = sign
		'''
		return s

			
			
class TraceWith(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self, name, sign, 
		operator = 0.0, # the operator with which the trace is taken
		dim = (), # dimension of input state
		**kwargs):
		
		kwargs['name'] = name
		kwargs['sign'] = sign
		kwargs['adj_name'] = name
		
		if 'dims' not in kwargs:
			kwargs['dims'] = {'in': dim, 'out': 1}
		super().__init__(**kwargs)
		self.operator = operator
		
		


class Identity(Maps):
	"""
	maps x -> x 
	"""
	def __init__(self,  sign, dim, **kwargs):
		
		if 'dims' not in kwargs:
			kwargs['dims'] = {'in': dim, 'out': dim}
		kwargs['name'] = 'Id'
		kwargs['sign'] = sign
		
		super().__init__(**kwargs)
		self.dim = dim
			

class Trace(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self,  sign, 
		dim = (),  # dimension of input state
		**kwargs):
		if 'dims' not in kwargs:
			kwargs['dims'] = {'in': dim, 'out': 1}
		if 'adj_name' not in kwargs:
			kwargs['adj_name'] = '1*'
		
		kwargs['name'] = 'Trace'
		kwargs['sign'] = sign
		
		super().__init__(**kwargs)
		self.dim = dim

class PartTrace(Maps):
	"""
	maps x -> trace_{subsystems}(x) ;  y -> id_{subsystems} \otimes y
	"""
	def __init__(self, sign, 
		subsystems,  # subsystems to trace : set
		state_dims, # tuple: dims of input
		**kwargs):
		
		kwargs['name'] = f"Trace_[{subsystems}/{len(state_dims)}]"
		kwargs['sign'] = sign
		
		remaining_subsystems = {i+1 for i in range(len(state_dims))}.difference(subsystems)
		
		if 'dims' not in kwargs:
			kwargs['dims'] =  {'in': np.prod(state_dims),
				'out': np.prod( [state_dims[i-1] for i in remaining_subsystems] ) 
				}
		
		if 'adj_name' not in kwargs:
			kwargs['adj_name'] = f'(x)Id_[{subsystems}/{len(state_dims)}]'

		
		super().__init__(**kwargs)
		self.subsystems = subsystems
		self.state_dims = state_dims
		
		
		
class CGmap(Maps):
	"""
	coare-graining map in kraus representation
	"""
	def __init__(self, name, sign,
		representation = None,  # the kraus representation of the map: a tuple of matrices
		action = {'dimsIn':(), 'pattern':(), 'dimsOut':()}, # action pattern: 
					# dimsIn: tuple  of dimensions of input satate
					# pattern: 
							# 0 means Id is acting on that system. 
							# integer values for blocks on which copies of the same map act:
							# (0,0,1,1,2,2) means IdxIdxCxC is applied where Id is acting on 1 spin and  C is acting on 2 spins
					# dimsOut: tuple of output dims
		**kwargs):
			
		if 'dims' not in kwargs:
			kwargs['dims'] = {'in': np.prod(action['dimsIn']), 'out': np.prod(action['dimsOut'])} # total dimensions
		if 'adj_name' not in kwargs:
			kwargs['adj_name'] = name + '^*'
		super().__init__(name, sign, **kwargs)
		self.representation = representation
		self.action = action
		
		
