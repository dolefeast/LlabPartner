#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, csv
import quantum_backend as quantum


def get_database(bits):
	database = []
	length = 2**bits
	for i in range(length):
		database.append(random.randint(1,800))
	database[0]=0
	return database

def import_ESI_data():
	with open("esi.csv",newline="") as file:
		data_reader = csv.reader(file,delimiter=",")
		data = [row for row in data_reader]
		del data[0]		# Remove Names/ESI line at the start of the file
		data_length = len(data)
		bits = int(np.ceil(np.log2(data_length)))
		entries = 2**bits
		for i in range(entries-data_length):	# Ensure the database length is a power of 2
			data.append(None)
	return data

def adaptive_oracle(x,x_0,database):
	Y = database[x_0][1]
	try:
		if database[x][1] > Y:
			return True
		else:
			return False
	except:
		return False

def adaptive_oracle2(x,x_0,database):
	Y = database[x_0]
	try:
		if database[x] < Y:
			return True
		else:
			return False
	except:
		return False

def adaptive_search(database,threshold):
	bits = int(np.ceil(np.log2(len(database))))
	while True:
		x_0 = random.randint(0,len(database)-1)
		if database[x_0] is not None:
			break
	scaling = 1.34
	m = 1
	fails = 0
	while fails < threshold:
		iterations = random.randint(1,np.ceil(m))
		J = quantum.Grover(lambda x: adaptive_oracle2(x,x_0,database),bits)
		q = J.search(iterations)
		x_1 = quantum.measure(q)
		if adaptive_oracle2(x_1,x_0,database):
			x_0 = x_1
			fails = 0
		else:
			fails += 1
		m = min(scaling*m,np.sqrt(2**bits))
	del J
	return x_0

def multi_trial_durr_hoyer(shots,trials,database,threshold):
	outputs = [[] for i in range(trials)] #también puedes hacer outputs = []*len(trials)
	bits = int(np.ceil(np.log2(len(database))))
	for j in range(trials):
		for i in range(shots):
			t = adaptive_search(database,threshold)
			outputs[j].append(t)
			print("Completed {}/{} shots, {}/{} trials".format(i+1,shots,j,trials),end="\r",flush=True)
	print("Completed {}/{} shots, {}/{} trials".format(shots,shots,trials,trials),end="\r",flush=True)	
	print("\nDone!")
	freq = [[0 for i in range(trials)] for j in range(2**bits)]
	for i in range(len(outputs)):
		for j in outputs[i]:
			freq[j][i] += 1
	return freq

def main1(shots,trials,bits):
    threshold = 9
    database = get_database(bits)

    freq = multi_trial_durr_hoyer(shots,trials,database,threshold)
    x = np.array([i for i in range(2**bits)])
    y,errors = [],[]
    for freq_list in freq:
        y.append(np.mean(freq_list))
        errors.append(np.std(freq_list))

    #true_target_pos = 636
    #true_target_pos = 0
    #fail_rate=100-y[true_target_pos]
    #fail_rate_error=errors[true_target_pos]
    #print("Fail rate: ({} +/- {})%".format(fail_rate,fail_rate_error))
    y = np.array(y)
    errors = np.array(errors)
    df = pd.DataFrame([x.T, y.T, errors.T]).T
    df.to_csv(f'./output/{shots}shots_{trials}trials_{bits}bits.csv', sep=',', index=False)


def plot_err(x, y, errors):
	plt.errorbar(x, y, yerr=errors, fmt="o", ecolor='gray', elinewidth=0.75, capsize=3)
	plt.xlabel("Database Register")
	plt.ylabel("Frequency")
	#plt.title(
	#	"""Frequency plot of average output of the Durr-Hoyer Algorithm
#for {} trials of {} shots""".format(trials,shots)
	#	)
	plt.title(
		"""Frequency plot of the Durr-Hoyer Algorithm for {} shots
when applied to the ESI database to find the most habitable planet""".format(shots)
		)
	plt.show()

def main2(shots,trials,bits):
	thresholds = [i for i in range(1,10+1)]
	database = get_database(bits)

	fail_rates, fail_rate_errors = [],[]
	for threshold in thresholds:
		print("Threshold:",threshold)
		freq = multi_trial_durr_hoyer(shots,trials,database,threshold)
		
		fail=100-np.mean(freq[0])
		fail_error=np.std(freq[0])
		print("Fail rate: ({} +/- {})%".format(fail,fail_error))
		fail_rates.append(fail)
		fail_rate_errors.append(fail_error)
	plt.errorbar(thresholds, fail_rates, yerr=fail_rate_errors, fmt="o", ecolor='gray', elinewidth=0.75, capsize=3)
	plt.xlabel("Termination Threshold")
	plt.ylabel("Failed Searches")
	plt.title(
		"""Mean number of failed shots of the Durr-Hoyer Algorithm
for {} trials of {} shots for different termination thresholds with {} qubits""".format(trials,shots,bits)
		)
	plt.show()
    
#print('Started first round!')
#print(main1(shots=100,trials=1,bits=10))
#print('Done!\n')
#print('Started second round!')
#print(main1(shots=100,trials=1,bits=2))
#print('Done!\n')
#print('Started third round!')
#print(main1(shots=300,trials=1,bits=10))
#print('Done!\n')
#print('Started fourth round!')
#print(main1(shots=100,trials=1,bits=3))
#print('Done!\n')
#print('Started fifth round!')
#print(main1(shots=100,trials=1,bits=5))
#print('Done!\n')
print('Started sixth round!')
main1(shots=700,trials=1,bits=10)
print('Done!\n')
print('Started seventh round!')
main1(shots=50,trials=1,bits=10)
print('Done!\n')
print('Started eigth round!')
main1(shots=10,trials=1,bits=10)
print('Done!\n')
print('Started nineth round!')
main1(shots=100,trials=2,bits=10)
print('Done!\n')
print('Started tenth round!')
main1(shots=100,trials=3,bits=10)
print('Done!\n')
print('Started eleventh round!')
main1(shots=100,trials=4,bits=10)
print('Done!\n')
print('Started twelveth round!')
main1(shots=100,trials=4,bits=11)
print('Done!\n')




