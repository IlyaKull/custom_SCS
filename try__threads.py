import os
import sys

os.environ["OMP_NUM_THREADS"] = sys.argv[1]

from multiprocessing.pool import ThreadPool
import numpy as np
import time


N = int(sys.argv[2])
n = int(sys.argv[3])

def print_arr(array):
	for a in array:
		print(f"In: {a[0]}")
		print(f"Out: {a[1]}")
	print('==================')


def initialize():
	array = []
	for i in range(N):
		mat = np.random.rand(n,n)
		mat = mat + mat.T
		array.append([mat, np.zeros((n,n)) ] )
	return array



def dgnlz(a):
	
	eigenvals, U =  np.linalg.eigh( a[0] )
	a[1] = U @ np.diag(eigenvals) @ U.T 
	# a[1] += a[0] @ a[0] @ a[0]
	# print(f"eigenvals: {eigenvals}")




array = initialize()
t = time.perf_counter()
# print_arr(array)
for a in array:
	dgnlz(a)
t = time.perf_counter() -t 
print(f"OMP num threads = {os.environ['OMP_NUM_THREADS']}; num workers = 1: time/iter = {t/N:0.3g}")

OMP_threads_per_worker = int(8 / int(os.environ['OMP_NUM_THREADS']))

for num_threads in range(2,OMP_threads_per_worker+1):
	array = initialize()

	t = time.perf_counter()

	with ThreadPool(processes=num_threads) as pool:
		pool.map(dgnlz, array)

	t = time.perf_counter() -t 
	print(f"OMP num threads = {os.environ['OMP_NUM_THREADS']}; num workers = {num_threads}: time/iter = {t/N:0.3g}")
# print_arr(array)
