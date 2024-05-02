
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
	
	C_l = maps.CGmap('C_l', +1)
	C_r = maps.CGmap('C_r', +1)
	
	tr_l_rho = maps.PartTrace(subsystems = ((1),dims_rho), sign = +1)
	tr_r_rho = maps.PartTrace(subsystems = ((5),dims_rho), sign = -1)
	
	tr_l_omega = maps.PartTrace(subsystems = ((1),dims_omega), sign = -1)
	tr_r_omega = maps.PartTrace(subsystems = ((3),dims_omega), sign = -1)
	
	tr = maps.Trace(+1, dim = np.prod(dims_rho) )
	
			
	Constraint('norm', (tr,), (rho,), constant = -1)
	Constraint('LTI', (tr_l_rho, tr_r_rho), (rho,rho))
	Constraint('left', (C_l, tr_l_omega), (rho,omega))
	Constraint('right', (C_r, tr_r_omega), (rho,omega)).print_constr_list()

	
	
	rho.print_var_list()
	
	
    

if __name__ == '__main__':
	main()
