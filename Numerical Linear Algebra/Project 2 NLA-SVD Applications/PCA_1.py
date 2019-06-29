import pandas as pd
import numpy as np
import scipy.linalg as sp

def rank(A,tol):
    u, s, v = np.linalg.svd(A, full_matrices=False)

    r = len(np.where(s > tol)[0])
    return(r)
	
def PCA_covariance(file):
    
    if file.endswith(".dat"):
        data = np.loadtxt(file)
        X = data.T
    elif file.endswith(".csv"):
        data = pd.read_csv(file,sep = ',')
        data = data.drop('gene', 1)
        X = data.values
    
    #mean of the observations for each variable
    mean_X = np.mean(X,axis=1) 
    
    n = X.shape[0]
    
    X_centered = X-mean_X[:,np.newaxis]
    Y = (1/np.sqrt(n-1))*X_centered.T

    U, s, VT= sp.svd(Y, full_matrices=False)

    tol = 1e-9
    r = rank(Y,tol)

    #1.Proportion of the variance accumulated in each principal component
    eigenvals = s**2
    sum_eigenvals = sum(eigenvals)
    prop_var = eigenvals/sum_eigenvals*100

    #2.Standard deviation of each principal component
    V = VT.T
    std_PCA = np.std(V,axis=1)

    for i in range(0,len(eigenvals)):
        print('Portion of total variance in the principle component', i+1, ': ', prop_var[i],'%')
    
        print('Standard deviation of principal component',i+1,': ',std_PCA[i],'\n')
    
    #3.Original dataset in the new PCA coordinates
    new_X = VT@X_centered

    for i in range(0,new_X.shape[1]):
        print('New PCA coordinates for observation ',i+1,': ',new_X[:,i])

		

def PCA_correlation(file):
    
    if file.endswith(".dat"):
        data = np.loadtxt(file)
        X = data.T
    elif file.endswith(".csv"):
        data = pd.read_csv(file,sep = ',')
        data = data.drop('gene', 1)
        X = data.values
    
    #mean of the observations for each variable
    mean_X = np.mean(X,axis=1)
    
    n = X.shape[0]

    X_centered = X-mean_X[:,np.newaxis]
    
    x_std = np.std(X,axis=1)
    
    X_std = X_centered/x_std[:,np.newaxis]

    Y = (1/np.sqrt(n-1))*X_std.T

    U, s, VT= sp.svd(Y)

    tol = 1e-9
    r = rank(Y,tol)

    #1.Proportion of the variance accumulated in each principal component
    eigenvals = s**2
    sum_eigenvals = sum(eigenvals)
    prop_var = eigenvals/sum_eigenvals*100

    #2.Standard deviation of each principal component
    V = VT.T
    std_PCA = np.std(V,axis=1)

    for i in range(0,len(eigenvals)):
        print('Portion of total variance in the principle component', i+1, ': ', prop_var[i],'%')
    
        print('Standard deviation of principal component',i+1,': ',std_PCA[i],'\n')
    
    #3.Original dataset in the new PCA coordinates
    new_X = VT@X_std

    for i in range(0,new_X.shape[1]):
        print('New PCA coordinates for observation ',i+1,': ',new_X[:,i])
 
print('\n--------------------------------------------------\n')
print('Perform PCA analysis using covariance matrix')
print('\n--------------------------------------------------\n')
PCA_covariance('example.dat')

print('\n--------------------------------------------------\n')
print('Perform PCA analysis using correlation matrix')
print('\n--------------------------------------------------\n')
PCA_correlation('example.dat')