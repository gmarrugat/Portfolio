import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)

def rule_3_4_variance(file):
    
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

	tol = 1e-9
	r = rank(Y,tol)

    #1.Proportion of the variance accumulated in each principal component
	eigenvals = s**2
	sum_eigenvals = sum(eigenvals)
	prop_var = eigenvals/sum_eigenvals*100
	
	var_total = sum(prop_var)
    
	for i in range(len(prop_var)):
        
		if sum(prop_var[:i]) >= (3/4)*var_total:
            
			num_PC = i
            
			print('Number of Principal Components needed: ',num_PC)
            
			break

print('\n--------------------------------------------------\n')
print('How many components are needed to explain the dataset under the 3/4 total variance rule?')
print('\n--------------------------------------------------\n')
rule_3_4_variance('RCsGoff.csv')