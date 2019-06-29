import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)


def LS_QR_fact_full_rank(file):
    

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
        
    Q,R = sp.qr(A)
    R1 = R[:r,:r]
    y = np.dot(Q.T,b)
    c = y[:r]
    
    x_qr = sp.solve_triangular(R1,c)
    x_qr_norm = np.linalg.norm(x_qr,2)
    lse_qr = np.linalg.norm(np.dot(A,x_qr) - b,2)
    
    print('The solution to the Least Square Problem using QR-Factorization for full rank matrices is:\n',x_qr)
    print('Its norm is:',x_qr_norm)
    print('and the Least Square Error obtained is:',lse_qr)
	
LS_QR_fact_full_rank('datafile.txt')