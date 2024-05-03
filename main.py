
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
	
		
	tr_l_omega = maps.PartTrace(subsystems = {1}, state_dims = dims_omega, sign = -1)
	tr_r_omega = maps.PartTrace(subsystems = {3}, state_dims = dims_omega, sign = -1)
	
	tr = maps.Trace(+1, dim = np.prod(dims_rho) )
	
	# dual varsiables
	a = OptVar('alpha', 'dual', dims = (d,d,d,d) )
	b_l = OptVar('beta_l', 'dual', dims = (D,d))
	b_r = OptVar('beta_r', 'dual', dims = (d,D))
	e = OptVar('epsilon', 'dual', dims = (1,))
	e.print_var_list()
	
	
	# primal constraints
	Constraint('norm', (tr,), (rho,), 'primal', 'eq', constant = -1, conjugateVar = e)
	Constraint('LTI', (tr_l_rho, tr_r_rho), (rho,rho), 'primal', 'eq', conjugateVar = a)
	Constraint('left', (C_l, tr_l_omega), (rho,omega), 'primal', 'eq', conjugateVar = b_l)
	Constraint('right', (C_r, tr_r_omega), (rho,omega), 'primal', 'eq', conjugateVar = b_r).print_constr_list()
	
	# dual constraints
	# Constraint('1', (C_r, tr_r_omega), (rho,omega), 'primal', 'eq', conjugateVar = b_r)
	
	
	

	
	
	
    

if __name__ == '__main__':
	main()
