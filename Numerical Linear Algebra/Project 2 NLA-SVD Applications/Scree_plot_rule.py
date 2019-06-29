import pandas as pd
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)

def Scree_plot_rule(file):

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
    
	factors = np.array(range(len(s))) + 1
    
	plt.figure(figsize=(8, 4))
	plt.plot(factors,s,'o')
	plt.xlabel('Number of Principal Components')
	plt.ylabel('Eigenvalue')
	plt.savefig('Scree_plot.png')
	
	
print('\n--------------------------------------------------\n')
print('How many components are needed to explain the dataset under the Scree plot rule?')
print('\n--------------------------------------------------\n')
Scree_plot_rule('RCsGoff.csv')