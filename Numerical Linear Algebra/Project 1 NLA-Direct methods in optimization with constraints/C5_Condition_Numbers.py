import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor,lu_solve, ldl, solve_triangular

def KKT_matrix(G,C,s,l,A=None):
    
    if A is None:
        
        col_1 = np.concatenate((G,-C.T,np.zeros([C.shape[1],C.shape[0]])),axis=0)
        col_2 = np.concatenate((-C,np.zeros([C.shape[1],C.shape[1]]),np.diag(s)),axis=0)
        col_3 = np.concatenate((np.zeros([C.shape[0],np.diag(l).shape[1]]),np.identity(np.diag(l).shape[1]),np.diag(l)),axis=0)

        KKT_matrix = np.concatenate([col_1,col_2,col_3],axis=1)
    
    else:
        col_1 = np.concatenate((G,-A.T,-C.T,np.zeros([C.shape[1],C.shape[0]])),axis=0)
        col_2 = np.concatenate((-A,np.zeros([A.shape[1],A.shape[1]]),np.zeros([C.shape[1],A.shape[1]]),np.zeros([C.shape[1],A.shape[1]])),axis=0)
        col_3 = np.concatenate((-C,np.zeros([A.shape[1],C.shape[1]]),np.zeros([C.shape[1],C.shape[1]]),np.diag(s)),axis=0)
        col_4 = np.concatenate((np.zeros([C.shape[0],np.diag(l).shape[1]]),np.zeros([A.shape[1],np.diag(l).shape[1]]),np.identity(np.diag(l).shape[1]),np.diag(l)),axis=0)
        
        KKT_matrix = np.concatenate([col_1,col_2,col_3,col_4],axis=1)
    
    return KKT_matrix

	
def condition_number_gral_1():
    # Initialization test problem
    A_file = 'opt_pr1/A.dad'
    C_file = 'opt_pr1/C.dad'
    G_file = 'opt_pr1/G.dad'

    n = 100
    m = 2*n
    p = 50

    A = np.zeros([n,p])

    A_coord = open(A_file,'r')

    for l in A_coord:
        row = l.split()
        A[int(row[0])-1,int(row[1])-1] = row[2]

    G = np.zeros([n,n])

    G_coord = open(G_file,'r')

    for l in G_coord:
        row = l.split()
        G[int(row[0])-1,int(row[1])-1] = row[2]
        G[int(row[1])-1,int(row[0])-1] = row[2]

    C = np.zeros([n,m])

    C_coord = open(C_file,'r')

    for l in C_coord:
        row = l.split()
        C[int(row[0])-1,int(row[1])-1] = row[2]

    s0 = lambda0 = np.ones(m)
    gamma0 = np.ones(p)

    M_KKT = KKT_matrix(G,C,s0,lambda0,A)

    K = np.linalg.norm(M_KKT,2)*np.linalg.norm(np.linalg.inv(M_KKT),2)
    
    return K

def condition_number_gral_2():
    # Initialization test problem
    A_file = 'opt_pr2/A.dad'
    C_file = 'opt_pr2/C.dad'
    G_file = 'opt_pr2/G.dad'

    n = 1000
    m = 2*n
    p = 500

    A = np.zeros([n,p])

    A_coord = open(A_file,'r')

    for l in A_coord:
        row = l.split()
        A[int(row[0])-1,int(row[1])-1] = row[2]

    G = np.zeros([n,n])

    G_coord = open(G_file,'r')

    for l in G_coord:
        row = l.split()
        G[int(row[0])-1,int(row[1])-1] = row[2]
        G[int(row[1])-1,int(row[0])-1] = row[2]

    C = np.zeros([n,m])

    C_coord = open(C_file,'r')

    for l in C_coord:
        row = l.split()
        C[int(row[0])-1,int(row[1])-1] = row[2]

    s0 = lambda0 = np.ones(m)
    gamma0 = np.ones(p)

    M_KKT = KKT_matrix(G,C,s0,lambda0,A)

    K = np.linalg.norm(M_KKT,2)*np.linalg.norm(np.linalg.inv(M_KKT),2)
    
    return K
	
print('The condition number for the problem 1 is ', condition_number_gral_1())
print('The condition number for the problem 2 is ', condition_number_gral_2())