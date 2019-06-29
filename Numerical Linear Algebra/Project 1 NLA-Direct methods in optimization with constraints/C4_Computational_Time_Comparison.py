import numpy as np
from scipy.linalg import lu_factor,lu_solve, ldl, solve_triangular
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
	
def KKT_matrix_s1(G,C,S,L_inv):
    col_1 = np.concatenate((G,-C.T))
    col_2 = np.concatenate((-C,-np.dot(L_inv,S)))

    KKT_matrix = np.concatenate((col_1,col_2),axis=1)
    
    return(KKT_matrix)
	
def Computational_Time(n):
    
    time_start = time.clock()
    # Initialization test problem
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

        d_z = np.linalg.solve(KKT,-r)

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

        d_z = np.linalg.solve(KKT,-r_hat)

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
	
def Computational_Time_with_LU(n):
    
    time_start = time.clock()
    # Initialization test problem
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
	
def Computational_Time_LDL(n):
    
    time_start = time.clock()

    # Initialization test problem
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
        L_inv = np.diag(1/lambda_i)

        # 1. Predictor substep

        r = F(x_i,s_i,lambda_i,g,d,G,C)

        r_l = r[:n]
        r_c = r[n:n+m]
        r_s = r[n+m:]

        #Solve the  KKT system with strategy 1
        KKT_s1 = KKT_matrix_s1(G,C,S,L_inv)
        r_s1 = np.concatenate((r_l,r_c - np.dot(L_inv,r_s)))

        #LDLT factorization
        L_s1, D_s1, perm = ldl(KKT_s1, lower = True)

        y = solve_triangular(L_s1,-r_s1, lower = True)

        d_z = solve_triangular(np.dot(D_s1,L_s1.T),y, lower = False)

        d_x = d_z[:n] 
        d_lambda = d_z[n:n+m]
        d_s = np.dot(L_inv,(-r_s-np.dot(S,d_lambda)))

        d_z = np.concatenate((d_z,d_s))

        # 2. Step-size correction

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

        r_hat_s1 = np.concatenate((r_l,r_c - np.dot(L_inv,r_s_hat)))

        y = solve_triangular(L_s1,-r_hat_s1, lower = True)

        d_z = solve_triangular(np.dot(D_s1,L_s1.T),y, lower = False)

        d_x = d_z[:n] 
        d_lambda = d_z[n:n+m]
        d_s = np.dot(L_inv,(-r_s_hat-np.dot(S,d_lambda)))

        d_z = np.concatenate((d_z,d_s))

        # 5. Step-size correction substep

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
	
def Computational_Time_Cholesky(n):
    
    time_start = time.clock()
    
    # Initialization test problem
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
        S_inv = np.diag(1/s_i)
        L = np.diag(lambda_i)

        # 1. Predictor substep

        r = F(x_i,s_i,lambda_i,g,d,G,C)

        r_l = r[:n]
        r_c = r[n:n+m]
        r_s = r[n+m:]

        G_hat = G + np.dot(C,np.dot(S_inv,np.dot(L,C.T)))
        r_hat = np.dot(-C,np.dot(S_inv,-r_s+np.dot(L,r_c)))
        #Cholesky factorization
        L_ch = np.linalg.cholesky(G_hat)

        y = solve_triangular(L_ch,-r_l - r_hat,lower=True)
        d_x = solve_triangular(L_ch.T,y,lower=False)
        d_lambda = np.dot(S_inv,-r_s + np.dot(L,r_c)) - np.dot(S_inv,np.dot(L,np.dot(C.T,d_x)))
        d_s = -r_c + np.dot(C.T,d_x)
        d_z = np.concatenate((d_x,np.concatenate((d_lambda,d_s))))

        # 2. Step-size correction

        alpha = Newton_step(lambda_i,d_lambda,s_i,d_s)

        # 3. Computation

        mu = np.dot(s_i.T,lambda_i)/m
        mu_hat = np.dot((s_i+alpha*d_s).T,(lambda_i+alpha*d_lambda))/m
        rho = (mu_hat/mu)**3

        # 4. Corrector substep
        r_l_hat = r[:n] 
        r_c_hat = r[n:n+m] 
        r_s_hat = r[n+m:] + np.dot(np.dot(np.diag(d_s),np.diag(d_lambda)),e) - rho*mu*e

        r_hat = np.dot(-C,np.dot(S_inv,-r_s_hat+np.dot(L,r_c_hat)))

        y = solve_triangular(L_ch,-r_l_hat - r_hat,lower=True)
        d_x = solve_triangular(L_ch.T,y,lower=False)
        d_lambda = np.dot(S_inv,-r_s_hat + np.dot(L,r_c_hat)) - np.dot(S_inv,np.dot(L,np.dot(C.T,d_x)))
        d_s = -r_c_hat + np.dot(C.T,d_x)
        d_z = np.concatenate((d_x,np.concatenate((d_lambda,d_s))))

        # 5. Step-size correction substep

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

compt_time_list = []
compt_time_list_with_LU = []
compt_time_list_with_LDL = []
compt_time_list_with_Cholesky = []

for i in n_list:
	total_time = Computational_Time(i)
	total_time_with_LU = Computational_Time_with_LU(i)
	total_time_with_LDL = Computational_Time_LDL(i)
	total_time_with_Cholesky = Computational_Time_Cholesky(i)
    
	compt_time_list.append(total_time)
	compt_time_list_with_LU.append(total_time_with_LU)
	compt_time_list_with_LDL.append(total_time_with_LDL)
	compt_time_list_with_Cholesky.append(total_time_with_Cholesky)
	
compt_time_list = np.array(compt_time_list)
compt_time_list_with_LU = np.array(compt_time_list_with_LU)
compt_time_list_with_LDL = np.array(compt_time_list_with_LDL)
compt_time_list_with_Cholesky = np.array(compt_time_list_with_Cholesky)


fig1 = plt.figure()
plot1 = fig1.add_subplot(111)
plot1.plot(n_list,compt_time_list,'red')
plot1.plot(n_list,compt_time_list_with_LDL,'green')
plot1.plot(n_list,compt_time_list_with_Cholesky,'yellow')
plot1.set_title('Computational time')
plot1.set_ylabel('Total time (milisec.)')
plot1.set_xlabel('Value of n')
plot1.legend(('General','LDL Factorization','Cholesky Factorization'), loc=2)
fig1.savefig('Computational_Time_Comparison_4_1.jpeg')


fig2 = plt.figure()
plot2 = fig2.add_subplot(111)
plot2.plot(n_list,compt_time_list_with_LU,'red')
plot2.plot(n_list,compt_time_list_with_LDL,'green')
plot2.plot(n_list,compt_time_list_with_Cholesky,'yellow')
plot2.set_title('Computational time')
plot2.set_ylabel('Total time (milisec.)')
plot2.set_xlabel('Value of n')
plot2.legend(('LU Factorization','LDL Factorization','Cholesky Factorization'), loc=2)
plt.show()
fig2.savefig('Computational_Time_Comparison_4_2.jpeg')