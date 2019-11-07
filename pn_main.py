"""
author: Fabian Schaipp
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from pn_helper import calculate_network,eps_estimation, clustering,draw_network, get_subgraph_by_cluster


    
X = pd.DataFrame( zscore(X0), columns = X0.columns, index=X0.index)

S = X[['Total Dribbles', 'Possession Gain',  'Possession Loss' , 'Interception Ratio', 'Total Shots', 'Short Pass Pct', 'Pass Accuracy' , 'Key Pass per Pass']]

#%% sort S after surnames
fullnames = list(S.index.values)
names = [f.split(" ")[-1].strip()  for f in fullnames]
S.loc[:, 'names'] = names
S = S.sort_values('names').drop('names', axis = 1)

# make sure the list names is in same order as index of S
names.sort()

#%%
metric = 'cosine'

G, D, edge_weights, edge_matrix = calculate_network(S, method = 'nearest neighbors' , metric = metric , gamma = None, k = 3 )


min_samples = 2 * len(S.columns)
eps_estimation(D, min_samples , metric)

clusters = clustering(D , eps = 0.45 , min_samples = min_samples, names = names)
    
# names have to have the same order as the node numbering in G!!
# this is ensured in calculate_network and by sorting S before hand
draw_network(G, names, clusters, edge_weights, write_labels = False, style = 'kamada')


# draw network for each sub-cluster
for c in np.unique(clusters):
    
    G1, edge_subset = get_subgraph_by_cluster(G, clusters, single_cluster = c)
    draw_network(G1, names, clusters.values[G1.nodes], edge_weights[edge_subset], write_labels = True, style = 'kamada')

  


