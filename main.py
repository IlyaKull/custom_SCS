
import maps
from variables import OptVar
from constraints import Constraint 
import numpy as np

def main():
	
	d = 2
	D = 3
	dims_rho = (d,d,d,d,d)
	dims_omega = (d,D,d)
	
	rho = OptVar('rho','primal', dims = dims_rho )
	omega = OptVar('omega','primal', dims = dims_omega)
	
	action_l = {'dimsIn': dims_rho, 'pattern':(1,1,1,1,0), 'dimsOut':(D,d)}
	action_r = {'dimsIn': dims_rho, 'pattern':(0,1,1,1,1), 'dimsOut':(d,D)}
	C_l = maps.CGmap('C_l', +1, action = action_l )
	C_r = maps.CGmap('C_r', +1, action = action_r )
	
	tr_l_rho = maps.PartTrace(subsystems = {1}, state_dims = dims_rho, sign = +1)
	tr_r_rho = maps.PartTrace(subsystems = {5}, state_dims = dims_rho, sign = -1)
	
		
	tr_l_omega = maps.PartTrace(subsystems = {1}, state_dims = dims_omega, sign = +1)
	tr_r_omega = maps.PartTrace(subsystems = {3}, state_dims = dims_omega, sign = -1)
	
	tr = maps.Trace(+1, dim = np.prod(dims_rho) )
	
	one_var = OptVar('1','primal',dims = (1), add_to_var_list = False)
	id_map = maps.Identity(+1)
	
	H_map = maps.TraceWith( 'H', +1, operator = [2,22,222] ,dim = np.prod(dims_rho) )
	
	# dual varsiables
	a = OptVar('alpha', 'dual', dims = (d,d,d,d) )
	b_l = OptVar('beta_l', 'dual', dims = (D,d))
	b_r = OptVar('beta_r', 'dual', dims = (d,D))
	e = OptVar('epsilon', 'dual', dims = (1,))
	
	
	
	# primal constraints
	Constraint('P_obj', (H_map,), (rho,), 'primal', 'OBJ')
	Constraint('pos_rho', (id_map,), (rho,), 'primal', 'PSD')
	Constraint('pos_rho', (id_map,), (omega,), 'primal', 'PSD')
	Constraint('norm', (tr,), (rho,), 'primal', 'EQ', constant = -1, conjugateVar = e)
	Constraint('LTI', (tr_l_rho, tr_r_rho), (rho,rho), 'primal', 'EQ', conjugateVar = a)
	Constraint('left', (C_l, tr_l_omega), (rho,omega), 'primal', 'EQ', conjugateVar = b_l)
	Constraint('right', (C_r, tr_r_omega), (rho,omega), 'primal', 'EQ', conjugateVar = b_r).print_constr_list()
	
	e.print_var_list()
	# dual constraints
	Constraint('D_obj', (id_map,), (e,), 'dual', 'OBJ')
	
	if False:
		map_list = [m for m in [H_map, C_l, C_r, tr_l_rho, tr_r_rho, id_map.mod_map(sign = -1) ]]
		
		for m in map_list:
			print( f"map name = {m.name} \nmap dims= {m.dims['out']},{m.dims['in']}" )
		
		map_list = [m.mod_map(adjoint = True) for m in [H_map, C_l, C_r, tr_l_rho, tr_r_rho, id_map.mod_map(sign = -1) ]]
		
		for m in map_list:
			print( f"map name = {m.name} \nmap dims= {m.dims['out']},{m.dims['in']}" )
	
	Constraint('1', [m.mod_map(adjoint = True) for m in [H_map, C_l, C_r, tr_l_rho, tr_r_rho, id_map.mod_map(sign = -1) ]], [one_var, b_l, b_r, a, a, e] , 'dual', 'PSD', conjugateVar = rho)
	
	
	

	
	
	
    

if __name__ == '__main__':
	main()
