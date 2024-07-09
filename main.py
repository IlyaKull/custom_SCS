import line_profiler

import numpy as np
from scs_funcs import SCS_Solver
import scs_funcs
from  variables import OptVar
from scipy.sparse.linalg import LinearOperator
import maps
import constraints
import matrix_aux_functions

import relax_LTI_N_problem


def main():
	
	# profile = line_profiler.LineProfiler()
	
	# profile.add_function(LinOp_id_plus_AT_A._matvec)
	# profile.add_function(scs_funcs.apply_dual_constr)
	# profile.add_function(scs_funcs.apply_primal_constr)
	# profile.enable_by_count()

	rng = np.random.default_rng(seed=17)
	
	relax_LTI_N_problem.set_problem(n=20, D=3, d=2, xOtimesI_impl = 'kron', cg_impl = 'kron')

	c = rng.random((OptVar.len_dual_vec_x,))
	b = rng.random((OptVar.len_primal_vec_y,))
	
	scs_solver = SCS_Solver(c, b)

	cc = c.copy()
	bc = b.copy()
	scs_solver.solve_M_inv(cc, bc)
	
	print(max(abs(cc-scs_solver.Minv_h[scs_solver.x_slice])), max(abs(bc-scs_solver.Minv_h[scs_solver.y_slice])) )
	
	  
if __name__ == '__main__':
	main()
	
