import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)
	
def is_full_rank(file):
    
    if file.endswith('.txt'):
        
        data = pd.read_csv(file,sep='  ', engine='python' ,header=None)
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
        
    elif file.endswith('.csv'):
        
        data = pd.read_csv(file,sep=',',header=None)
        m = data.shape[0]
        n = data.shape[1]-1
        A = data.iloc[:,:n]
        b = data.iloc[:,-1]
        A = np.array(A)
        
    
    tol = 1e-9
    r = rank(A,tol)
    
    if r == A.shape[1]:
        print('Yes, matrix is full rank')
    elif r < A.shape[1]:
        print('No, matrix is not full rank, it is rank deficient')
		
print('\n--------------------------------------------------\n')
print('datafile.txt - Is the matrix of the dataset datafile.txt full rank?')
print('\n--------------------------------------------------\n')
is_full_rank('datafile.txt')
