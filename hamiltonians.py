import numpy as np
import math

kron = np.kron

def make_hamiltonian_term(model):
	
	match model['name']:
		case 'AKLT':
			d=3; 
			Sx = math.sqrt(2)/2 * np.array([ [0, 1, 0], [1, 0, 1], [0, 1, 0]])
			iSy = -1*math.sqrt(2)/2 * np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]])
			Sz = np.diag([1,0,-1])
			S_dot_S = kron(Sx,Sx) - kron(iSy,iSy) + kron(Sz,Sz)
			h_term = 0.5*( S_dot_S + (1/3) * S_dot_S @ S_dot_S + (2/3) * np.identity(d**2) );
		
			return h_term, d
		
		# case 'XY_in_transverse_field' % eq 19 in https://scipost.org/SciPostPhysCore.4.4.027
			# d=2;  
			# X = [0 1; 1 0]; Y = 1i*[0 -1; 1 0]; Z = [1 0; 0 -1];
			# r = model.r;
			# Jx = ((r + inv(r))^2)/2;
			# Jy = ((r - inv(r))^2)/2;
			# B1 = r^2 - inv(r^2);
			# epsilon = r^2 + inv(r^2);
			
			# h_term_n = @(n) -Jx * tensor(X,speye(d^(n-2)),X) + ...
							   # -Jy *real( tensor(Y,speye(d^(n-2)),Y) )+ ...
							   # +B1/2 * (tensor(Z,speye(d^(n-2)),speye(d)) + tensor(speye(d),speye(d^(n-2)),Z) )+ ...
							   # epsilon *speye(d^n) ;
		# case 'ANNNI' % eq 22 in https://scipost.org/SciPostPhysCore.4.4.027
			# d=2;  
			# X = [0 1; 1 0]; Y = 1i*[0 -1; 1 0]; Z = [1 0; 0 -1];
			# r = model.r;
			# Jz = ((r - inv(r))^2)/4;
			# B2 = (r^2 - inv(r^2))/2;
			# epsilon = ((r + inv(r))^2)/4;
			
			# h_term_n = @(n)  -1* tensor(X,speye(d^(n-2)),X) + ...
							  # Jz * tensor(Z,speye(d^(n-2)),Z) + ...
							  # B2/2 * (tensor(Z,speye(d^(n-2)),speye(d)) + tensor(speye(d),speye(d^(n-2)),Z) )+ ...
							  # epsilon *speye(d^n) ;
		# case 'q_deformed_XXZ' % eq 38 in https://scipost.org/SciPostPhysCore.4.4.027
			# d=2;  
			# X = [0 1; 1 0]; Y = 1i*[0 -1; 1 0]; Z = [1 0; 0 -1];
			# q = model.q;
					
			# h_term_n = @(n)  -q/4*(...
							  # tensor(X,speye(d^(n-2)),X) + ...
							  # tensor(Y,speye(d^(n-2)),Y) + ...
							  # (q+inv(q))/2 * (tensor(Z,speye(d^(n-2)),Z) - speye(d^n) ) + ...
							  # (q-inv(q))/2 * (tensor(Z,speye(d^(n-2)),speye(d)) - tensor(speye(d),speye(d^(n-2)),Z)) ...
							  # ) ;
		# case 'Z3-ANNNP'
			# d=3;
			# r= model.r;
			# sigma = [0 1 0; 0 0 1; 1 0 0];
			# omega= exp(2i*pi/3);
			# tau = [1 0 0; 0 omega 0; 0 0 omega^2];
			
			# f = 2*(1+2*r)*(1-r^3)/9/(r^2); % eq 96-97 in https://scipost.org/SciPostPhysCore.4.4.027
			# epsilon = 2*((1+r+r^2)^2)/9/(r^2);
			# g1 = -2*((1-r)^2)*(1+r+r^2)/9/(r^2);
			# g2 = ((1-r)^2)*(1-2*r-2*(r^2))/9/(r^2);
			
			# temp_term = @(n) tensor(sigma,speye(d^(n-2)),sigma') + ...
							  # f/2* ( tensor(speye(d),speye(d^(n-2)),tau) + tensor(tau,speye(d^(n-2)),speye(d)))+...
							  # g1 * tensor(tau,speye(d^(n-2)),tau) + ...
							  # g2 * tensor(tau,speye(d^(n-2)),tau') ;
						  
			# h_term_n = @(n)  -1*(temp_term(n) + temp_term(n)' ) + speye(d^n)*epsilon ;
			
			
			# case 'Z3-XY'
			# d=3;
			# r= model.r;
			# sigma = [0 1 0; 0 0 1; 1 0 0];
			# omega= exp(2i*pi/3);
			# tau = [1 0 0; 0 omega 0; 0 0 omega^2];
			
			# b = (r^3-1)/(r^3+2); % eq 79-80 (with p=3) in https://scipost.org/SciPostPhysCore.4.4.027
			# f = 6*(1-r^6)/((r^3+2)^2); 
			# epsilon = 6*(r^6+2)/((r^3+2)^2); % typo in eq 80. 6*(...) instead of 3*(...) see eq 85
			
			# one_plus_bSumTau = speye(d) + b*(tau + tau^2);
			# term_to_conj = @(n) tensor(one_plus_bSumTau * sigma',speye(d^(n-2)),sigma * one_plus_bSumTau ) ; 
			# h_term_n = @(n)  speye(d^n)*epsilon  -real(...
								# term_to_conj(n) + term_to_conj(n)' + ...
								# f/2* (   tensor(tau,speye(d^(n-2)),speye(d)) + tensor(speye(d),speye(d^(n-2)),tau)  ) + ...
								# f/2* (   tensor(tau^2,speye(d^(n-2)),speye(d)) + tensor(speye(d),speye(d^(n-2)),tau^2)  ) ...
								# );
				
