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

    
    return new_X, VT, prop_var

print('\n--------------------------------------------------\n')
print('Perform PCA analysis using covariance matrix')
print('\n--------------------------------------------------\n')
new_X, PC, portion_var  = PCA_covariance('RCsGoff.csv')	

data = pd.read_csv('RCsGoff.csv',sep = ',')
data = data.drop('gene', 1)
index = data.columns
num = np.array(range(1,len(data.columns)+1))
PCs = ['PC'+str(n)for n in num]

new_X_coord = pd.DataFrame(data = new_X.T,index = index, columns = PCs)
var_prop_df = pd.DataFrame(data = portion_var.T, index = PCs,columns = ['portion_variance'])
output_df = pd.concat([new_X_coord,var_prop_df.T])

output_df.to_csv('PCA_dataset2.csv', sep = ',')
