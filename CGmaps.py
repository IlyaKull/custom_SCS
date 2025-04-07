import numpy as np
from ncon import ncon
import util

import logging
logger = logging.getLogger(__name__)


def main():
			
	Din = 2
	Dout = 2
	U = np.array([
					[1, 0, 1, 0],
					[0, 1, 0, 1],
					[1, 0, -1, 0],
					[0, 1, 0, -1],
				]
	)
	
	
	kraus = cptp_from_unitary_w_gvec( Din, Dout, U)
	for j in range(len(kraus)):
		print(f'kraus op {j}\n {kraus[j]}')
	
	
	## full test for two layers of cptp cg maps
	Din = 2
	Dout = 2
	V = util.random_unitary(Din**2)
	
	kraus = cptp_from_unitary_w_gvec( Din, Dout, V)
	for j in range(len(kraus)):
		print(f'kraus op {j}\n {kraus[j]}')
	
	
	
	
def test_iso_from_MPS(V0, L, R, V, d, k0, n):
	# the left and right isometries should satisfy:
	# L_k *(I,V_k)  = V_k+1  ;    R_k *(V_k,I)  = V_k+1
	for k in range(k0,n):
		print(f'testing k = {k}', end= '... ')
		try:
			assert np.allclose(V[k+1], L[k] @ np.kron(np.identity(d), V[k])), '!! left'
			assert np.allclose(V[k+1], R[k] @ np.kron(V[k], np.identity(d))), '!! right'
		except AssertionError:
			print(f'k={k}')
			raise
		else:
			print('passed')




def iso_from_MPS(MPS, k0, n, return_all_Vs=False):
	'''
	input MPS as 3-dim array
	'''
	logger.info(f'computing isometries, k0 = {k0}, n = {n}')
	nmax=24
	try:
		assert n<=nmax
	except AssertionError:
		logger.critical(f'isometries can be computed with the current method (QR) up to nmax = {nmax}, n = {n}')
		raise
	
	d = MPS.shape[1]
	D = MPS.shape[0]
	assert D == MPS.shape[2], f'MPS shape problem: MPS.shape = {MPS.shape}'
	
	P = [[],]*(n+1)
	
	L = [[],]*(n+1)
	R = [[],]*(n+1)
	if return_all_Vs:
		V = [[],]*(n+1)
	
	for k in range(k0,n+1):
		logger.debug(f'k={k} isometry started')
		indCell=[[-(k+1), -1, 1],];
		for l in range(1,k-1):
			indCell.append( [l, -(l+1), l+1])
		indCell.append([(k-1), -(k), -(k+2)])
		
		'''
		compute the tensor composed of k mps tensors
		contarction pattern: 
							(-k-1) --A-(1)-A-(2)-A-(3)-A-(4)-A-- (-k-2)
									 |     |     |     |     |
									 -1    -2   -3    -4    -5
		
		 because we want 
						   (-k-2)--VV--(-5)
								   VV--(-4)
								   VV--(-3)
								   VV--(-2)
						   (-k-1)--VV--(-1)
		'''
		# print(f'k={k}')
		# print(f'indCell = {indCell}')
		# print(f'P = {P}')
		MPSProd = ncon([MPS,]*k, indCell, forder=[-k-1, -k-2] + list(range(-1,-k-1,-1)));
		MPSProdMat = np.reshape(MPSProd,(D**2,d**k))
		
		# to obtain an isometry from  MPSProdMat   P^-1 needs to be applied to it
		# where P^2 is the following D^2xD^2 matrix:
		
		Q,r = np.linalg.qr(MPSProdMat.T)
		P[k]=r.T
		
		if k==k0:
			V0 = Q.T
		if return_all_Vs:
			V[k] = Q.T
			
		logger.debug(f'k={k} isometry done')
	'''
	Left and Right Isometries
	the left and right isometries should satisfy:
	L_k *(I,V_k)  = V_k+1  ;    R_k *(V_k,I)  = V_k+1 

	   --LL--VV--   --VV--     ;   --RR------   --VV-- 
		 LL  VV--     VV--     ;     RR--VV--     VV-- 
		 LL  VV--  =  VV--     ;     RR  VV--  =  VV-- 
		 LL--VV--     VV--     ;     RR  VV--     VV-- 
	   --LL------   --VV--     ;   --RR--VV--   --VV-- 

	 i.e.
		 L_k = (P_k+1)^-1 * (PmpsL)_k

	   --LL--     --Pinv-------PP--    %   --RR----    --Pinv   .-------
		 LL         Pinv       PP      %     RR--        Pinv   |   PP--      
		 LL     =   Pinv--mmm--PP      %     RR     =    Pinv--www--PP      
		 LL--       Pinv   |   PP--    %     RR          Pinv       PP      
	   --LL----   --Pinv   .-------    %   --RR--      --Pinv-------PP--     

	 where  P is r' from the qr decomposition and we define (PmpsL)_k and (PmpsR)_k :

		-------PP--      --PmpsL--    ;     .-------        PmpsL---
			   PP          PmpsL      ;     |   PP--        PmpsL--
		--mps--PP     =: --PmpsL      ;  --www--PP     =: --PmpsL
		   |   PP--        PmpsL--    ;         PP          PmpsL
		   .-------        PmpsL---   ;  -------PP--      --PmpsL--
	 
	 (mmm) and (www) stand for the mps tensor  :

								 (d) 
	   (l)--mmm--(r)              |  
			 |         =    (r)--www--(l)
			(d)                
	 as we don't want to compute r^-1 explicitly
	 we solve  
	   (r_k+1) * L_k = (PmpsL)_k 
	 for L_k, and similarly for R_k:
	   (r_k+1) * R_k = (PmpsR)_k


	  compute PmpsL and PmpsR
	 PmpsL:
	
	 (-1)------(1)-PP--(-3)     
	 (-1)--mps-(1)-PP--(-3)
	        |           
	        .----------(-2)       
	
	 PmpsR:
	        .----------(-3)
	        |
	 (-1)--www-(1)-PP--(-2)     
	 (-1)------(1)-PP--(-2)           
	'''
	
	PmpsL = [[],]*(n+1)
	PmpsR = [[],]*(n+1)
		
	mpsInflatedR = np.kron(np.expand_dims(np.identity(D),1), MPS)
	mpsInflatedL = np.kron(MPS, np.expand_dims(np.identity(D),1))
	
	# # # # print(f'mps[:,0,:] = \n{MPS[:,0,:]}')
	# # # # print(f'INFLmps_L[:,0,:] = \n{mpsInflatedL[:,0,:]}')
	# # # # print(f'INFLmps_R[:,0,:] = \n{mpsInflatedR[:,0,:]}')
	
	          
	for k in range(k0,n+1):     
		# compute PL{k} and PR{k}
		PmpsL[k] = ncon([mpsInflatedL, P[k]], [[-1,-2,1], [1,-3]]).reshape((D**2, D*D*d) )  
		PmpsR[k] = ncon([mpsInflatedR, P[k]], [[1,-3,-1], [1,-2]]).reshape((D**2, D*D*d) )   



	## compute left and right isometries
	# recall that L and R are defined by P{k+1}*L{k} = MPS*r{k} =:PmpsL{k}
	# ie  P{k+1}*L{k} = PL{k}
	# and P{k+1}*R{k} = PR{k}
	# we solve these equations for L and R 
	 
	for k in range(k0,n):
		L[k] = np.linalg.solve(P[k+1], PmpsL[k])	   
		R[k] = np.linalg.solve(P[k+1], PmpsR[k])  
	
	logger.info('done')
	
	if return_all_Vs:
		return V0, L, R, V
	else:
		return V0, L, R
	



def plain_cg_from_MPS(MPS, k0, n):
	'''
	input MPS as 3-dim array
	'''
	logger.info(f'making plain cg maps, k0 = {k0}, n = {n}')
		
	
	d = MPS.shape[1]
	D = MPS.shape[0]
	assert D == MPS.shape[2], f'MPS shape problem: MPS.shape = {MPS.shape}'
	
	# transform into left gauge
	mps_as_map_to_left = MPS.reshape((d*D,D))
	Q,r = np.linalg.qr(mps_as_map_to_left, mode='reduced')
	MPS = Q.reshape((D,d,D))
	
	
	P = [[],]*(n+1)
	
	L = [[],]*(n+1)
	R = [[],]*(n+1)
	
	
	
	indCell=[[-(k0+1), -1, 1],];
	for l in range(1,k0-1):
		indCell.append( [l, -(l+1), l+1])
	indCell.append([(k0-1), -(k0), -(k0+2)])
	
	'''
	compute the tensor composed of k mps tensors
	contarction pattern: 
						(-k-1) --A-(1)-A-(2)-A-(3)-A-(4)-A-- (-k-2)
								 |     |     |     |     |
								 -1    -2   -3    -4    -5
	
	 because we want 
					   (-k-2)--VV--(-5)
							   VV--(-4)
							   VV--(-3)
							   VV--(-2)
					   (-k-1)--VV--(-1)
	'''
	# print(f'k={k}')
	# print(f'indCell = {indCell}')
	# print(f'P = {P}')
	MPSProd = ncon([MPS,]*k0, indCell, forder=[-k0-1, -k0-2] + list(range(-1,-k0-1,-1)));
	V0 = np.reshape(MPSProd,(D**2,d**k0))
	

	'''
	Left and Right Isometries
	 

	   --LL--      -------        %   --RR----        .---    
		 LL                       %     RR--          |             
		 LL     =  --mmm--        %     RR     =   --www--        
		 LL--         |           %     RR                        
	   --LL----       .---        %   --RR--       -------         

	  
	 (mmm) and (www) stand for the mps tensor  :

								 (d) 
	   (l)--mmm--(r)              |  
			 |         =    (r)--www--(l)
			(d)                

	
	 (-1)------(1)-PP--(-3)     
	 (-1)--mps-(1)-PP--(-3)
	        |           
	        .----------(-2)       
	
	 PmpsR:
	        .----------(-3)
	        |
	 (-1)--www-(1)-PP--(-2)     
	 (-1)------(1)-PP--(-2)           
	'''
	mpsInflatedR = np.kron(np.expand_dims(np.identity(D),1), MPS)
	mpsInflatedL = np.kron(MPS, np.expand_dims(np.identity(D),1))
	
	# when computing the isometreis we had
	# PmpsL[k] = ncon([mpsInflatedL, P[k]], [[-1,-2,1], [1,-3]]).reshape((D**2, D*D*d) )  
	# PmpsR[k] = ncon([mpsInflatedR, P[k]], [[1,-3,-1], [1,-2]]).reshape((D**2, D*D*d) )   
	# here we dont have the Ps
	       
	mpsL = ncon([mpsInflatedL, ], [[-1,-2,-3], ]).reshape((D**2, D*D*d) )  
	mpsR = ncon([mpsInflatedR, ], [[-2,-3,-1], ]).reshape((D**2, D*D*d) )   
	
	for k in range(k0,n):
		L[k] = mpsL
		R[k] = mpsR

	
	logger.info('done')
	
	return V0, L, R
		
	



def cptp_from_unitary(Din, Dout, U):
	'''
	make kraus operators of dims (Dout+1, Din**2) from unitary U of dims (Din^2 x Din^2).
	The first Dout rows of U are taken to make the first kraus operator
	An additional vector is added to the output dimension
	kraus[0] = [U(1, ...),
				U(2, ...), 
				...
				U(Dout, ...), 
				[zeros ] ]  <--- additional garbage collection vector
				
	the other kraus operators are then rank one operators mapping the remaining rows of U to the additional vector
	
	'''
	
	assert U.shape == (Din**2, Din**2)
	assert Dout <= Din**2
	#   Dout+1  Dout+1
	#    |      |
	#   U       U     
	#   /\      /\
	#  /  \    /  \
	#  o  o    o  o   
	#Din  Din 
	
	
	kraus = []
	kraus.append( np.vstack([ U[:Dout, ...],
							  np.zeros((1,Din**2))
							]) 
				)
	for j in range(Dout,Din**2):
		kraus.append( np.vstack([ 	np.zeros((Dout,Din**2)),
									U[j, ...]									
								]) 
					)
					
	return kraus
	


def cptp_from_unitary_w_gvec(Din, Dout, U):
	'''
	same as above, only now we map from the product two spaces of dims Din+1, into Dout+1
	the +1 dim is the garbage collectin vector. 
	U is Din^2 x Din^2.
	When we map from (Din+1) x (Din+1), whenever the input has support in either of the garbage vectors, we map it to the garbage of the output. 
	e.g. (see test in main() func above) if 
	Din = 2
	Dout = 2
	U = np.array([
					[1, 0, 1, 0],
					[0, 1, 0, 1],
					[1, 0, -1, 0],
					[0, 1, 0, -1],
				]
	of the 9 vectors in the input space, the following are mapped into the garbage vector of the output: 
	[0  1  2*,
	 3  4  5*, 
	 6* 7* 8*]
	The remaining ones are "work indices". in the "work space we make the first kraus op from the first two rows of U. 
	Then every further row of U gets mapped to the output garbage. 
	Finally, the garbage inds are also mapped to garbage.
	
				
	'''
	assert U.shape == (Din**2, Din**2)
	assert Dout <= Din**2
	#   Dout+1  Dout+1
	#    |      |
	#   U       U     
	#   /\      /\
	#  /  \    /  \
	#  o  o    o  o   
	#Din+1 Din+1 
	
	inds = np.arange((Din+1)**2).reshape((Din+1,Din+1))
	g_inds = list(set(inds[-1,...]).union( set(inds[...,-1]) ))
	w_inds = list( set(inds.ravel()).difference(set(g_inds)) )
	g_inds.sort()
	# print(f' g inds: {g_inds}')
	w_inds.sort()
	# print(f' w inds: {w_inds}')
	
	
	kraus = []
	kraus.append( np.zeros( (Dout+1, (Din+1)**2) )	)
	kraus[0][:Dout, w_inds] = U[:Dout, ...]
	
	
	for j in range(Dout,Din**2):
		k= np.zeros( (Dout+1, (Din+1)**2)) 
		k[Dout, w_inds] = U[j,...]
		kraus.append(k)
					
	for j in range(len(g_inds)):
		k= np.zeros( (Dout+1, (Din+1)**2)) 
		k[Dout, g_inds[j]] = 1
		kraus.append(k)
		
	return kraus
	




	
if __name__ == '__main__':
	main()
