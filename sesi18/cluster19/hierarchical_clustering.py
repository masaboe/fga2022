# Hierarchical Clustering

# Importing the libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('covid19_cfr.csv')
X = dataset.iloc[:, [1,3,4,5,6,7,8,9,10,11]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Covid Cases')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'C 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'C 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'brown', label = 'C 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'yellow', label = 'C 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'black', label = 'C 5')
plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 100, c = 'purple', label = 'C 6')



plt.title('Clusters of province')
plt.xlabel('Covid Cases in Indonesia (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Evaluation
from sklearn.metrics import davies_bouldin_score
w=davies_bouldin_score(X, y_hc)
print("Score davies bouldin ")
print(w)

#Cluster result
dfLabels=pd.DataFrame(y_hc, columns=["hasil"])+1
Cluster = 'Cluster ' + pd.DataFrame(dfLabels['hasil'].map(str) + ' - ' + dataset['provinsi'].map(str), columns=["result"]).sort_values(by=["result"])