import numpy as np
from scipy.linalg import lu_factor,lu_solve, ldl, solve_triangular

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
	
def f(x,g,G):
    
    return (1/2)*np.dot(np.dot(x.T,G),x)+np.dot(g.T,x)
	
def KKT_matrix_s1_v2(G,C,A,s,l):
    col_1 = np.concatenate((G,-A.T,-C.T))
    col_2 = np.concatenate((-A,np.zeros([A.shape[1],A.shape[1]]),np.zeros([C.shape[1],A.shape[1]])))
    col_3 = np.concatenate((-C,np.zeros([A.shape[1],C.shape[1]]),-np.dot(np.linalg.inv(np.diag(l)),np.diag(s))))

    KKT_matrix = np.concatenate((col_1,col_2,col_3),axis=1)
    
    return(KKT_matrix)
	
# Initialization test problem
A_file = 'opt_pr1/A.dad'
b_file = 'opt_pr1/b.dad'
C_file = 'opt_pr1/C.dad'
d_file = 'opt_pr1/d.dad'
G_file = 'opt_pr1/G.dad'
g_file = 'opt_pr1/g_.dad'

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
    #G is symmetric
    G[int(row[1])-1,int(row[0])-1] = row[2]

C = np.zeros([n,m])

C_coord = open(C_file,'r')

for l in C_coord:
    row = l.split()
    C[int(row[0])-1,int(row[1])-1] = row[2]

b = np.zeros(p)

b_coord = open(b_file,'r')

for l in b_coord:
    row = l.split()
    b[int(row[0])-1] = row[1]

d = np.zeros(m)

d_coord = open(d_file,'r')

for l in d_coord:
    row = l.split()
    d[int(row[0])-1] = row[1]

g = np.zeros(n)

g_coord = open(g_file,'r')

for l in g_coord:
    row = l.split()
    g[int(row[0])-1] = row[1]
    

x0 = np.zeros(n)
s0 = lambda0 = np.ones(m)
gamma0 = np.ones(p)


e = np.ones(m)
epsilon = 1e-16
max_iter = 100

z_list = []
z = np.concatenate((x0,gamma0,lambda0,s0))

iterations = 0

while True:
    
    #z
    x_i = z[:n] 
    gamma_i = z[n:n+p]
    lambda_i = z[n+p:n+p+m]
    s_i = z[n+p+m:]
    S = np.diag(s_i)
    L = np.diag(lambda_i)
    
    # 1. Predictor substep
    KKT = KKT_matrix(G,C,s_i,lambda_i,A)

    r = F(x_i,s_i,lambda_i,g,d,G,C,A,gamma_i,b)
    
    r_l = r[:n]
    r_g = r[n:n+p]
    r_c = r[n+p:n+p+m]
    r_s = r[n+p+m:]
    
     #Solve the  KKT system with strategy 1
    KKT_s1 = KKT_matrix_s1_v2(G,C,A,s_i,lambda_i)
    r_s1 = np.concatenate((r_l,r_g,r_c - np.dot(np.linalg.inv(np.diag(lambda_i)),r_s)))

    #LDLT factorization
    L_s1, D_s1, perm = ldl(KKT_s1, lower = True)

    y = solve_triangular(L_s1,-r_s1, lower = True)

    d_z = solve_triangular(np.dot(D_s1,L_s1.T),y, lower = False)

    # 2. Step-size correction

    d_x = d_z[:n] 
    d_gamma = d_z[n:n+p]
    d_lambda = d_z[n+p:n+p+m]
    d_s = np.dot(np.linalg.inv(np.diag(lambda_i)),(-r_s-np.dot(S,d_lambda)))
    d_z = np.concatenate((d_z,d_s))

    alpha = Newton_step(lambda_i,d_lambda,s_i,d_s)

    # 3. Computation

    mu = np.dot(s_i.T,lambda_i)/m
    mu_hat = np.dot((s_i+alpha*d_s).T,(lambda_i+alpha*d_lambda))/m
    rho = (mu_hat/mu)**3

    # 4. Corrector substep
    r_l_hat = r[:n] 
    r_g_hat = r[n:n+p]
    r_c_hat = r[n+p:n+p+m] 
    r_s_hat = r[n+p+m:] + np.dot(np.dot(np.diag(d_s),np.diag(d_lambda)),e) - rho*mu*e
    
    r_hat_s1 = np.concatenate((r_l_hat,r_g_hat,r_c_hat - np.dot(np.linalg.inv(np.diag(lambda_i)),r_s_hat)))

    y = solve_triangular(L_s1,-r_hat_s1, lower = True)

    d_z = solve_triangular(np.dot(D_s1,L_s1.T),y, lower = False)


    # 5. Step-size correction substep

    d_x = d_z[:n] 
    d_gamma = d_z[n:n+p]
    d_lambda = d_z[n+p:n+p+m]
    d_s = np.dot(np.linalg.inv(np.diag(lambda_i)),(-r_s_hat-np.dot(S,d_lambda)))
    
    d_z = np.concatenate((d_z,d_s))

    alpha = Newton_step(lambda_i,d_lambda,s_i,d_s)

    # 6. Update substep

    z_next = z + 0.95*alpha*d_z
    z_list.append(z_next)
    
    z = z_next
    
    iterations = iterations + 1
    
    if(all(np.absolute(r_l_hat)<epsilon) or all(np.absolute(r_c_hat)<epsilon) or np.absolute(mu)<epsilon or iterations>max_iter):
        print('Stop criterion because:')
        if(all(np.absolute(r_l_hat)<epsilon) or all(np.absolute(r_c_hat)<epsilon) or np.absolute(mu)<epsilon):
            print('Function f is close enough to its minimum. The norm of the gradient of f at the solution found is', np.linalg.norm(F(z[:n],z[n+p+m:],z[n+p:n+p+m],g,d,G,C,A,z[n:n+p],b),2))
            print('f(x)=',f(x_i,g,G))
            break
        if(iterations>max_iter):
            print('Maximum number of iterations have been exceeded')
            break
print('then, we have reached the solution of the problem 1')
print('\nIterations:', iterations)