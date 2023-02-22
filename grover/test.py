import numpy as np
import matplotlib.pyplot as plt
import random, time
import quantum_backend as quantum

def main():
	bits = 2
	database = []
	length = 2**bits
	for i in range(length):
		database.append(0)
	targets=1
	for i in range(targets):
		database[0]=1

	def oracle_function(x):
		if database[x] == 1:
			return True
		else:
			return False
			
	print(len(database))
	
	iterations = int(np.ceil(np.sqrt(len(database)/targets)))
	print(iterations)
	start_time=time.time()
	J = quantum.Grover(oracle_function,bits,verbose=True)
	T=[]
	shots = 1024
	t = J.search(iterations)
	print("Starting shots!")
	for i in range(shots):
		T.append(quantum.measure(t))
		print("Completed {}/{} shots".format(i+1,shots),end="\r",flush=True)
	print("\nDone!")
	readings = []
	freq = []
	for i in range(2**bits):
		readings.append(i)
		freq.append(0)
	for i in T:
		freq[i] += 1
	x = np.array(readings)
	y = np.array(freq)

	plt.scatter(x,y)
	plt.xlabel("Database Register")
	plt.ylabel("Frequency")
	end_time=time.time()
	print("Runtime: {}s".format(end_time-start_time))
	del J
	plt.show()

if __name__=="__main__":
	main()