This is a so-called matrix-free implementation of the SCS algorithm [https://doi.org/10.1007/s10957-016-0892-3], specialized for solving problems of the form described in  https://arxiv.org/abs/2212.03014.

The idea is that the SDP to be solved involves several or many positive semidefinite variables $\{\rho_i\}$ of modest size that are constrained pairwise as $C_i\rho_i == M_i \rho_{i+1}$, where $C_i$ and $M_i$ are some linear maps.
If the total problem size becomes too large to form the full constraints matrix directly and one is forced to rsort to the indirect SCS method, it is much more efficient to implement the the maps $M_i, C_i$ as functions, rather than as matices acting on the vectorized $\rho_i$.
E.g., if the maps $M_i$ are partial trace maps (as in https://arxiv.org/abs/2212.03014), then implementing the map $\rho \mapsto \mathrm{Tr}_{s}\rho$, where $s$ is the set of subsystems to be traced, as a function that sums up the appropriate entries of $\rho$ is more efficient than vectorizing $\rho$ and applying the matrix representation of the partial trace map.

In this implementation optimization variables, maps, constraints etc. are implemented as classes, such that one can define and modify their behaviour as appropriate for a specific problem.
Depending on the dimensions of the optimization variables, certain ways of implementing a given map can be more efficient than others.
For example if  $\rho$ is a state on systems $1,2,3$ and the  map $C$ to be applied to it acts only on a subset of the systems, say $2,3$, one could implement $\rho \mapsto C\rho$ either by index contraction (reshape, matrix multiply and reshape, e.g., np.einsum or equivalent), or one could first construct the map $\mathbb{I}\otimes C$ and apply that on the whole of $\rho$. The 'maps' objects thus have an '.implementation' attribute which controls which implementation is used.  

Several example problems are specified in the modules LTI_N_problem.py, GAP_LTI_N_problem.py, and relax_LTI_N_problem.py. To run a specific problem, the corresponding module has to be chosen as the problem_module in main.py. 
In addition to global command line args specified in main.py, each problem has its own problem-cpecific  settings which can be specified as commnad-line arguments. The latter are defined in the corresponding problem module. 


