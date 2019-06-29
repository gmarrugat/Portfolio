import numpy as np
from scipy.linalg import ldl, solve_triangular
import sys

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
	
def KKT_matrix_s1(G,C,s,l):

    col_1 = np.concatenate((G,-C.T))
    col_2 = np.concatenate((-C,-np.dot(np.linalg.inv(np.diag(l)),np.diag(s))))

    KKT_matrix = np.concatenate((col_1,col_2),axis=1)
    
    return(KKT_matrix)
	

# Initialization test problem
n = int(sys.argv[1])
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
    S = np.diag(s_i)
    L = np.diag(lambda_i)
    
    # 1. Predictor substep
    KKT = KKT_matrix(G,C,s_i,lambda_i)

    r = F(x_i,s_i,lambda_i,g,d,G,C)
    
    r_l = r[:n]
    r_c = r[n:n+m]
    r_s = r[n+m:]

    #Solve the  KKT system with strategy 1
    KKT_s1 = KKT_matrix_s1(G,C,s_i,lambda_i)
    r_s1 = np.concatenate((r_l,r_c - np.dot(np.linalg.inv(np.diag(lambda_i)),r_s)))

    #LDLT factorization
    L_s1, D_s1, perm = ldl(KKT_s1, lower = True)

    y = solve_triangular(L_s1,-r_s1, lower = True)

    d_z = solve_triangular(np.dot(D_s1,L_s1.T),y, lower = True)
    
    d_x = d_z[:n] 
    d_lambda = d_z[n:n+m]
    d_s = np.dot(np.linalg.inv(np.diag(lambda_i)),(-r_s-np.dot(S,d_lambda)))
    
    d_z = np.concatenate((d_z,d_s))

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
    
    r_hat_s1 = np.concatenate((r_l_hat,r_c_hat - np.dot(np.linalg.inv(np.diag(lambda_i)),r_s_hat)))

    y = solve_triangular(L_s1,-r_hat_s1, lower = True)

    d_z = solve_triangular(np.dot(D_s1,L_s1.T),y, lower = False)
    
    d_x = d_z[:n] 
    d_lambda = d_z[n:n+m]
    d_s = np.dot(np.linalg.inv(np.diag(lambda_i)),(-r_s_hat-np.dot(S,d_lambda)))
    
    d_z = np.concatenate((d_z,d_s))

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
        print('Stop criterion because:')
        if(all(np.absolute(r_l_hat)<epsilon) or all(np.absolute(r_c_hat)<epsilon) or np.absolute(mu)<epsilon):
            print('Function f is close enough to its minimum. The norm of the gradient of f at the solution found is', np.linalg.norm(F(z[:n],z[n+m:],z[n:n+m],g,d,G,C),2))
            break
        if(iterations>max_iter):
            print('Maximum number of iterations have been exceeded')
            break
x_min = z[:n]
print('\nThe x minimum achieved is:',x_min)
print('which is equal to the value of -g (',-g,')')
print('then, we have reached the solution of the test problem')
print('\nIterations:', iterations)