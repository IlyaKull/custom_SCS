import numpy as np
from ncon import ncon


import logging
logger = logging.getLogger(__name__)


def main():
	D = 5
	d = 3
	k0=3
	n=7
	A = np.random.rand(D,d,D)
	
	V0, L, R, V = iso_from_MPS(A, k0, n, True)
	
	assert np.allclose(V[k0],V0)
	test_iso_from_MPS(V0, L, R, V, d, k0, n)
	
	
	
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
	

def plain_cg_from_MPS(MPS, k0, n):
	'''
	input MPS as 3-dim array
	'''
	logger.info(f'makin plain cg maps, k0 = {k0}, n = {n}')
	
	
	d = MPS.shape[1]
	D = MPS.shape[0]
	assert D == MPS.shape[2], f'MPS shape problem: MPS.shape = {MPS.shape}'
	
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
	 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
	





	
if __name__ == '__main__':
	main()
