import time
import numpy as np

	 
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


def random_unitary(d):
	x = np.random.normal(size=(d,d))
	q,r = np.linalg.qr(x)
	return q
	

class NullContextManager:
	def __init__(self, dummy_resource=None):
		self.dummy_resource = dummy_resource
	def __enter__(self):
		return self.dummy_resource
	def __exit__(self, *args):
		pass

class DummyExecutor:
	def __init__(self):
		pass
		
	def submit(self, func, *args, **kwargs):
		# calls the function on the input
		func(*args, **kwargs)
