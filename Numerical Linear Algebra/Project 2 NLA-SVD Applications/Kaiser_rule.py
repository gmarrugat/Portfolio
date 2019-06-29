import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)
	
def Kaiser_rule(file):

	if file.endswith(".dat"):
		data = np.loadtxt(file)
		X = data.T
	elif file.endswith(".csv"):
		data = pd.read_csv(file,sep = ',')
		data = data.drop('gene', 1)
		X = data.values
    
    #mean of the observations for each variable
	mean_X = np.mean(X,axis=1)
    
	n = X.shape[0]

	X_centered = X-mean_X[:,np.newaxis]

	Y = (1/np.sqrt(n-1))*X_centered.T

	U, s, VT= sp.svd(Y, full_matrices=False)
    
	num_PC = sum(s > 1)
    
	print('Number of Principal Components needed: ',num_PC)
	
print('\n--------------------------------------------------\n')
print('How many components are needed to explain the dataset under the Kaise rule?')
print('\n--------------------------------------------------\n')
Kaiser_rule('RCsGoff.csv')