"""
author: Fabian Schaipp
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from pn_helper import calculate_network,eps_estimation, clustering,draw_network, get_subgraph_by_cluster, plot_heatmap

"""
import your matrix with data as an array called X
rows should be players, columns should be the statistics
"""

def import_fbref_data():
    
    defensive = pd.read_csv('fbref/defensive.csv', index_col = 0)
    leagues = defensive.reindex(columns=['Squad', 'Comp']).set_index('Squad')
    defensive = defensive.drop(columns = ['Comp','# Pl']).set_index('Squad')
    
    passes = pd.read_csv('fbref/passes.csv', index_col = 0)
    passes = passes.drop(columns = ['Comp','# Pl']).set_index('Squad')
    
    possession = pd.read_csv('fbref/possession.csv', index_col = 0)
    possession = possession.drop(columns = ['Comp','# Pl']).set_index('Squad')
    
    shots = pd.read_csv('fbref/shots.csv', index_col = 0)
    shots = shots.drop(columns = ['Comp','# Pl']).set_index('Squad')
    
    X = pd.concat([defensive, passes, possession, shots], axis = 1)
    
    return X
    

X = import_fbref_data()

corr = np.corrcoef(X.T)
sns.heatmap(corr, xticklabels = X.columns, yticklabels = X.columns)

S = pd.DataFrame( zscore(X), columns = X.columns, index=X.index)

#%% sort S after surnames

fullnames = list(S.index.values)

# use one of these lines (depending on players --> first line, or teeam --> second line)
names = [f.split(" ")[-1].strip()  for f in fullnames]
names = fullnames.copy()

S.loc[:, 'names'] = names
S = S.sort_values('names').drop('names', axis = 1)

# make sure the list names is in same order as index of S
names.sort()

#%%
metric = 'cosine'

G, D, edge_weights, edge_matrix = calculate_network(S, method = 'nearest neighbors' , metric = metric , gamma = None, k = 3 )


min_samples = 2 * len(S.columns)
eps_estimation(D, min_samples , metric)

# set epsilon to where the kink/elbow lies in the graph above
eps = 0.1

clusters = clustering(D , eps = eps , min_samples = min_samples, names = names)
    
# names have to have the same order as the node numbering in G!!
# this is ensured in calculate_network and by sorting S before hand
draw_network(G, names, clusters, edge_weights, write_labels = True, style = 'kamada')


# draw network for each sub-cluster
for c in np.unique(clusters):
    
    G1, edge_subset = get_subgraph_by_cluster(G, clusters, single_cluster = c)
    draw_network(G1, names, clusters.values[G1.nodes], edge_weights[edge_subset], write_labels = True, style = 'kamada')

  

plot_heatmap(S)
