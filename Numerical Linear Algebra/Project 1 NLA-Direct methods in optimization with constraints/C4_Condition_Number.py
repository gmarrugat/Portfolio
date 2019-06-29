import numpy as np
import matplotlib.pyplot as plt

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

def condition_number_ineq(n):
    
    m = 2*n

    G = np.identity(n)
    C = np.concatenate((np.identity(n),-np.identity(n)),axis=1)

    s0 = lambda0 = np.ones(m)

    M_KKT = KKT_matrix(G,C,s0,lambda0)

    K = np.linalg.norm(M_KKT,2)*np.linalg.norm(np.linalg.inv(M_KKT),2)
    
    return(K)
	
	
K_list_ineq = []

n_list = np.arange(1,100,10)

for i in n_list:
    K = condition_number_ineq(i)
    K_list_ineq.append(K)
    
K_list_ineq = np.array(K_list_ineq)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(n_list,K_list_ineq,'b-')
ax.set_xlabel('Value of n')
ax.set_ylabel('Condition Number')
ax.set_title('Condition Number')
ax.set_ylim((0,5))
plt.show()
fig.savefig('Condition Number.jpeg')