

 
class Maps:
	"""
	
	"""
	
	def __init__(self, 
		name, # the name of the map in your paper notes. Used to for display.
		sign, # maps come with a sign (+/-1): determinig the sign with which they appear in a constraint
		):
		self.name = name
		self.sign = sign
		
		
class CGmap(Maps):
	"""
	coare-graining map in kraus representation
	"""
	def __init__(self, name, sign,
		representation = 0.0  # the kraus representation of the map: a tuple of matrices
		):
		super().__init__(name, sign)
		self.representation = representation
			
			
class TraceWith(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self, op_name, sign, 
		operator = 0.0 # the operator with which the trace is taken
		):
		name = f"{op_name}*"
		super().__init__(name, sign)
		self.operator = operator

class Trace(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self,  sign, 
		dim = ()  # dimension of input state
		):
		name = 'Trace'
		super().__init__(name, sign)
		self.dim = dim
			
class PartTrace(Maps):
	"""
	maps x -> trace_{subsystems}(x) ;  y -> id_{subsytems} \otimes y
	"""
	def __init__(self, sign, 
		subsystems  # subsytems to trace 
		):
		name = f"Trace_[{subsystems[0]}/{len(subsystems[1])}]"
		super().__init__(name, sign)
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
		
