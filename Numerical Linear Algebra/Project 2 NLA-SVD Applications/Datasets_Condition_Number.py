import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)

def condition_number(A):
    u, s, v = np.linalg.svd(A)
    norm_2 = s[0]
    tol = 1e-9
    norm_2_inv = s[rank(A,tol)-1]
    k = norm_2/norm_2_inv
    print('Condition number: ',k)
    return(k)
	
	
data = pd.read_csv('datafile.txt',sep='  ', engine='python' ,header=None)
m = data.shape[0]
n = data.shape[1]-1
x = data.iloc[:,:n]
b = data.iloc[:,-1]
        
# As we did in pr4 we compute a polynomial of order 3
# with the coeficients of the dataset datafile.txt
        
A = np.zeros(((x.shape[0]),3))

A[:,0] = 1
A[:,1] = x[0]
A[:,2] = x[0]**2

print('\n-----------------------------------------\n')
print('Condition number for dataset datafile.txt')
print('\n-----------------------------------------\n')
condition_number(A)

data = pd.read_csv('datafile2.csv',sep=',',header=None)
m = data.shape[0]
n = data.shape[1]-1
A = data.iloc[:,:n]
b = data.iloc[:,-1]
A = np.array(A)

print('\n-----------------------------------------\n')
print('Condition number for dataset datafile2.csv')
print('\n-----------------------------------------\n')
condition_number(A)