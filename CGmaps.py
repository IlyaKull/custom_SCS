

 
class CGmap:
	"""
	Represents a coarse-graining map. Each map has an attribute specifying the primal variable on 
	which it acts. Each map can an act on several such variables. The cgmap.representation attribute 
	stores the Kraus operators representing the map.
	"""
	
	def __init__(self, 
		name, # the name of the map in your paper notes
		sign, # maps come with a sign (+/-1): determinig the sign with which they appear in a constraint
		representation = 0.0 # the data representing the map: a tuple of matrices
		):
		self.name = name
		self.representation = representation
		self.sign = sign
		
			
			
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
		
