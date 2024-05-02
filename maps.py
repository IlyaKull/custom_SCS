
import numpy as np
 
class Maps:
	"""
	
	"""
	
	def __init__(self, 
		name, # the name of the map in your paper notes. Used to for display.
		sign, # maps come with a sign (+/-1): determinig the sign with which they appear in a constraint
		dims # input and output dimensions
		):
		self.name = name
		self.sign = sign
		self.dims = dims
		
		
class CGmap(Maps):
	"""
	coare-graining map in kraus representation
	"""
	def __init__(self, name, sign,
		representation = None,  # the kraus representation of the map: a tuple of matrices
		action = {'dimsIn':(), 'pattern':(), 'dimsOut':()} # action pattern: 
					# dimsIn: tuple  of dimensions of input satate
					# pattern: 
							# 0 means Id is acting on that system. 
							# integer values for blocks on which copies of the same map act:
							# (0,0,1,1,2,2) means IdxIdxCxC is applied where Id is acting on 1 spin and  C is acting on 2 spins
					# dimsOut: tuple of output dims
		):
		dims = {'in': np.prod(action['dimsIn']), 'out': np.prod(action['dimsOut'])} # total dimensions
		super().__init__(name, sign, dims)
		self.representation = representation
		self.action = action
		
			
			
class TraceWith(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self, op_name, sign, 
		operator = 0.0, # the operator with which the trace is taken
		dim = () # dimension of input state
		):
		super().__init__(f"{op_name}*", sign, dims = {'in': dim, 'out': 1})
		self.operator = operator
		

class Trace(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self,  sign, 
		dim = ()  # dimension of input state
		):
		super().__init__('Trace', sign, dims = {'in': dim, 'out': 1})
			

class PartTrace(Maps):
	"""
	maps x -> trace_{subsystems}(x) ;  y -> id_{subsystems} \otimes y
	"""
	def __init__(self, sign, 
		subsystems,  # subsystems to trace : set
		state_dims # tuple: dims of input
		):
		name = f"Trace_[{subsystems}/{len(state_dims)}]"
		remaining_subsystems = {i+1 for i in range(len(state_dims))}.difference(subsystems)
		dims = {'in': np.prod(state_dims),
				'out': np.prod( [state_dims[i-1] for i in remaining_subsystems] ) }
		super().__init__(name, sign, dims)
		self.subsystems = subsystems
		
		
class MPSmap(CGmap):
	"""
	Represnts a coarse-graining map in the MPS-based C-G scheme.
	Such maps are used in in the ground-state enrgy (GSE) problem and in the gap certification problem (GAP).
	In GSE the maps act on the leftmost or rightmost m spins of a state, i.e. either as IxW (right version) or as WxI (left version)
	In GAP there are two additional left untouched by the map, one on each side of the state. The same map W act in this setting as IxWxII (left) or  IIxWxI (right).
	"""	
	
	def __init__(self,
		representation, acts_on, sign, # as in parent class "CGmap"
		id_pad # the number of identity terms padding the map  (0 for GSE, 1 for GAP)
		):
		super().__init__(representation, acts_on)
		self.id_pad = id_pad 
		
		
	def apply(self, l_r, adj, var):
		"""
		apply map to variable
		l_r specifies left or right version
		adj specifies adjoint
		"""
		return 
		
