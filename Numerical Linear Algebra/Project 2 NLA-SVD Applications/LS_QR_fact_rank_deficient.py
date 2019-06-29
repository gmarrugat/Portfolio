import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)
	
	
def LS_QR_fact_rank_def(file):
    
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
    
    Q,R,P = sp.qr(A,pivoting=True)
    R1 = R[:r,:r]
    S = R[:r,r:m]
    
    sol_Qtb = np.dot(Q.T,b)
    c = sol_Qtb[:r]
    
    u = sp.solve_triangular(R1,c)
    
    v = np.zeros(n-r)
    sol_Ptx = np.append(u,v)
    
    x_qr_def = np.zeros(sol_Ptx.shape)
    x_qr_def[P] = sol_Ptx
    
    x_qr_norm = np.linalg.norm(x_qr_def,2)
    
    lse_qr_def = np.linalg.norm(np.dot(A,x_qr_def) - b,2)
    
    print('The solution to the Least Square Problem using QR-Factorization for rank deficient matrices is:\n',x_qr_def)
    print('Its norm is:',x_qr_norm)
    print('and the Least Square Error obtained is:',lse_qr_def)
	
LS_QR_fact_rank_def('datafile2.csv')