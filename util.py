import time
	 
def sign_str(x): 
	s = f"{x:+}"
	if abs(x) == 1:
		s = s[0]
	return s

def determine_k0(d,chi):
	
	try:
		k0 = next( k for k in range(100) if chi < d**k)	
	except StopIteration:
		print('Could not determine k0 value < 100' )
		raise
	
	return k0
