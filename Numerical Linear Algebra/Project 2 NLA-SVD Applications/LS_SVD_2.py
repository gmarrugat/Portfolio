import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)


def LS_SVD(file):
    
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
    
    u, s, vt = sp.svd(A)

    r = np.linalg.matrix_rank(A)
    S_pse_inv = np.zeros(A.shape).T
    S_pse_inv[:r,:r] = np.diag(1/s[:r])

    x_svd = vt.T.dot(S_pse_inv).dot(u.T).dot(b)

    x_svd_norm = np.linalg.norm(x_svd,2)
    lse_svd = np.linalg.norm(np.dot(A,x_svd) - b,2)

    print('The solution to the Least Square Problem using Singular Value Decomposition is:\n',x_svd)
    print('Its norm is:',x_svd_norm)
    print('and the Least Square Error obtained is:',lse_svd)
	
LS_SVD('datafile2.csv')