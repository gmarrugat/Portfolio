import numpy as np
from scipy.linalg import lu_factor,lu_solve
import sys
import time
import matplotlib.pyplot as plt

def Newton_step(lamb0,dlamb,s0,ds):
    alp=1
    idx_lamb0=np.array(np.where(dlamb<0))
    if idx_lamb0.size>0:
        alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size>0:
        alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    return alp
	
	
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

def F(x,s,l,g,d,G,C,A=None,gamma=None, b=None):
    
    if A is None:
        r_L = np.dot(G,x) + g - np.dot(C,l) 
        r_c = s + d - np.dot(C.T,x)
        r_s = s*l 
        r = np.concatenate((r_L, r_c, r_s))
        
    else:
        r_L = np.dot(G,x) + g - np.dot(A,gamma) - np.dot(C,l)
        r_Lam = b - np.dot(A.T,x) 
        r_c = s + d -np.dot(C.T,x)
        r_s = s*l 
        r = np.concatenate((r_L, r_Lam, r_c, r_s))
    
    return(r)
	
def Computational_Time_with_LU(n):
    
    time_start = time.clock()
    # Initialization test problem
    n = n
    m = 2*n

    G = np.identity(n)
    C = np.concatenate((np.identity(n),-np.identity(n)),axis=1)

    d = -10*np.ones(m)


    mu = 0
    sigma = 1
    g = np.random.normal(mu,sigma,n)

    x0 = np.zeros(n)
    s0 = lambda0 = np.ones(m)


    e = np.ones(m)
    epsilon = 1e-16
    max_iter = 100

    z_list = []
    z = np.append(x0,np.append(lambda0,s0))

    iterations = 0

    while True:

        #z
        x_i = z[:n] 
        lambda_i = z[n:n+m]
        s_i = z[n+m:]

        # 1. Predictor substep
        KKT = KKT_matrix(G,C,s_i,lambda_i)

        r = F(x_i,s_i,lambda_i,g,d,G,C)

        LU,P = lu_factor(KKT)

        d_z = lu_solve((LU,P),-r)

        # 2. Step-size correction

        d_x = d_z[:n] 
        d_lambda = d_z[n:n+m]
        d_s = d_z[n+m:]

        alpha = Newton_step(lambda_i,d_lambda,s_i,d_s)

        # 3. Computation

        mu = np.dot(s_i.T,lambda_i)/m
        mu_hat = np.dot((s_i+alpha*d_s).T,(lambda_i+alpha*d_lambda))/m
        rho = (mu_hat/mu)**3

        # 4. Corrector substep
        r_l_hat = r[:n] 
        r_c_hat = r[n:n+m] 
        r_s_hat = r[n+m:] + np.dot(np.dot(np.diag(d_s),np.diag(d_lambda)),e) - rho*mu*e

        r_hat = np.append(r_l_hat,np.append(r_c_hat,r_s_hat))

        d_z = lu_solve((LU,P),-r_hat)

        # 5. Step-size correction substep

        d_x = d_z[:n] 
        d_lambda = d_z[n:n+m]
        d_s = d_z[n+m:]

        alpha = Newton_step(lambda_i,d_lambda,s_i,d_s)

        # 6. Update substep

        z_next = z + 0.95*alpha*d_z
        z_list.append(z_next)

        z = z_next

        iterations = iterations + 1

        if(all(np.absolute(r_l_hat)<epsilon) or all(np.absolute(r_c_hat)<epsilon) or np.absolute(mu)<epsilon or iterations>max_iter):
            time_end = time.clock()
            computational_time = (time_end - time_start)*1000
            break

    return computational_time
	
n_list = np.arange(1,100)
compt_time_list_with_LU = []

for i in n_list:
    total_time_with_LU = Computational_Time_with_LU(i)
    compt_time_list_with_LU.append(total_time_with_LU)
compt_time_list_with_LU = np.array(compt_time_list_with_LU)

#see the cubic polynomial behaviour of the computational time
coef = np.polyfit(n_list, compt_time_list_with_LU, 3)
pol = np.poly1d(coef)
n = np.linspace(n_list[0],n_list[-1],100)


fig1 = plt.figure()
plot1 = fig1.add_subplot(111)
plot1.plot(n,pol(n),'k', linewidth=0.7, label="cubic polynomial")
plot1.plot(n_list,compt_time_list_with_LU,'blue')
plot1.set_title('Computational time with LU Factorization')
plot1.set_ylabel('Total time (milisec.)')
plot1.set_xlabel('Value of n')
plot1.legend(('Cubic Polynomial Function','Computational Time with LU'))
plt.show()
fig1.savefig('Computational_Time_with_LU.jpeg')