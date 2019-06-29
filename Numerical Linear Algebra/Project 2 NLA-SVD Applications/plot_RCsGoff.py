import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PCA_df = pd.read_csv('PCA_dataset2.csv', sep=',',index_col=0)
new_X = np.array(PCA_df.iloc[:20,:2])
plt.figure(figsize=(10, 5))
plt.plot(new_X[:,0], new_X[:,1], 'o')
plt.xlabel('PC1: 72.2991066629% variance')
plt.ylabel('PC2: 15.7844663293% variance')
plt.savefig('RCsGoff_PCA.png')