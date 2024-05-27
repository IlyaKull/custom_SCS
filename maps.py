
import numpy as np
import copy

from matrix_aux_functions import vec2symm
 
class Maps:
	"""
	
	"""
	
	def __init__(self, 
		name, # the name of the map in your paper notes. Used to for display.
		dims, # input and output dimensions
		adj = False, # adjoint flag: true to apply the adjoint
		adj_name = None # display name for adjoint operator
		): 
		self.name = name
		self.dims = dims
		self.adjoint_flag = adj
		self.adj_name = adj_name
		
	
	
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
		
		
	def __call__(self, in_var, out_var, in_vec, out_vec):
		""" 
		applies the map to the components of in_vec contatining in_var from the list of variables 
		and assigns the result to the components of out_vec contatining out_var. 
		e.g.
		constraint  = trace(rho*h) - 1 = 0 with conj bariable alpha
		and in_vec = [rho, sigma, omega], out_vec = [alpha, beta]
		then
		TraceWith: rho --> trace(h*rho) --> alpha component of out_vec
		"""
		

		
		assert (in_var.primal_or_dual == 'primal' and out_var.primal_or_dual == 'dual') or \
				(in_var.primal_or_dual == 'dual' and out_var.primal_or_dual == 'primal'), \  
				('in_var and out_var sould be a prima-dual pair\n.' + f'in_var {in_var.name} is {in_var.primal_or_dual}\n' + 
				f'out_var {out_var.name} in {out_var.primal_or_dual}' )
			)
		
		
		if np.prod(in_var.dims) == 1:
			matrix_var = vec2symm(dim = 1 , in_vec[in_var.indices[0]]) # real scalar
		else if in_var.complex:
			matrix_var = vec2symm(dim = np.prod(in_var.dims) ,  in_vec[in_var.indices[0]:in_var.indices[1]])
		else:
			matrix_var = vec2symm(dim = np.prod(in_var.dims) ,  real_part = in_vec[in_var.indices[0]:in_var.indices[1]],
				imag_part =in_vec[in_var.indices[2]:in_var.indices[3]]
			)
		
		
		
class CGmap(Maps):
	"""
	coare-graining map in kraus representation
	"""
	def __init__(self, name, 
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
		super().__init__(name, dims, adj_name = name + '^*')
		self.representation = representation
		self.action = action
		
			
			
class TraceWith(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self, op_name, 
		operator = 0.0, # the operator with which the trace is taken
		dim = () # dimension of input state
		):
		 		
		super().__init__(f"{op_name}*", dims = {'in': dim, 'out': 1}, adj_name = op_name)
		self.operator = operator
	
	
		
		
		
		


class Identity(Maps):
	"""
	maps x -> x 
	"""
	def __init__(self,  dim):
				
		super().__init__('Id', dims = {'in': dim, 'out': dim})
		
			

class Trace(Maps):
	"""
	maps x -> trace(O*x) ; its adjoint is then c -> c*O
	"""
	def __init__(self,  
		dim = ()  # dimension of input state
		):
		self._initial_dim = dim
		super().__init__('Trace', dims = {'in': dim, 'out': 1}, adj_name = '1*')
			

class PartTrace(Maps):
	"""
	maps x -> trace_{subsystems}(x) ;  y -> id_{subsystems} \otimes y
	"""
	def __init__(self,  
		subsystems,  # subsystems to trace : set
		state_dims # tuple: dims of input
		):
				
		name = f"Trace_[{subsystems}/{len(state_dims)}]"
		remaining_subsystems = {i+1 for i in range(len(state_dims))}.difference(subsystems)
		dims = {'in': np.prod(state_dims),
				'out': np.prod( [state_dims[i-1] for i in remaining_subsystems] ) }
		super().__init__(name, dims, adj_name = f'(x)Id_[{subsystems}/{len(state_dims)}]')
		self.subsystems = subsystems
		
		


	
	
	
