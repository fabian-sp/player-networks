"""
author: Fabian Schaipp
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics



def calculate_network(S, method = 'nearest neighbors' , metric = 'cosine', gamma = None, k = None ):
    if metric == 'cosine' and k == None:
        k = 3
        
    if metric == 'cosine' and gamma == None:
        gamma = 0.45

    
    D = metrics.pairwise_distances(S, metric = metric)
    
    
    if method == 'nearest neighbors':
        nbrs = NearestNeighbors(n_neighbors= k,  metric = metric).fit(S)
        
        distances, indices = nbrs.kneighbors(S)
        adj = nbrs.kneighbors_graph(S).toarray()
        
    elif method == 'threshold':
        adj = (D <= gamma).astype(int)
        
    else:
        raise ValueError("Not a known method")
    
    edge_matrix = pd.DataFrame(adj, columns = S.index.values, index = S.index).T
    G= nx.from_numpy_array(adj)
        
    # now get the weight(= distance) for each edge 
    edge_weights=[]
    
    if method == 'nearest neighbors':
        for e in list(G.edges):
            if np.argwhere(indices[e[0]] == e[1]).size > 0:
                col_num = np.argwhere(indices[e[0]] == e[1])
                row_num = e[0]
                
            else: 
                col_num = np.argwhere(indices[e[1]] == e[0])
                row_num = e[1]
        
            col_num = np.asscalar(np.squeeze(col_num))    
            edge_w = float(distances[row_num, col_num])
            edge_weights.append(edge_w)
    
    elif method == 'threshold':
        for e in list(G.edges):
            edge_weights.append( D[e[1], e[0]] )
    
    else:
        raise ValueError("Not a known method")    
    
    # we want high edge weights for small distances --> invert
    edge_weights = np.array(edge_weights)
    edge_weights = edge_weights.max() + 1 - edge_weights  
    
    return G, D, edge_weights, edge_matrix


def clustering(D , eps , min_samples, names):
    
    #n_neighbors_per_player = (D <= eps).sum(axis = 1)

    cluster_obj = DBSCAN(min_samples = min_samples, eps = eps, metric = 'precomputed').fit(D)

    # label - 1 is noise
    clusters = cluster_obj.labels_
    print ("Number of clusters: ", np.unique(clusters, return_counts = True))

    #cluster_info = pd.DataFrame( np.vstack((n_neighbors_per_player, clusters)).T ,  index = S.index, columns = ['Nneighbors', 'Label'])
    cluster_series = pd.Series(clusters, index = names)
    
    return cluster_series

def eps_estimation(D, min_samples , metric):
    
    nbrs_eps = NearestNeighbors(n_neighbors= min_samples ,  metric = metric).fit(D)
    distances_eps, _ = nbrs_eps.kneighbors(D)

    sns.set_style("dark")
 
    plt.figure()
    plt.plot(np.sort(distances_eps.max(axis = 1))[::-1])
    plt.xlabel('Rank')
    plt.ylabel('Distance')
    
    return

def get_subgraph_by_cluster(G, clusters, single_cluster):
    
    node_subset = list(np.array(G.nodes)[clusters == single_cluster])
    G1 = G.subgraph(node_subset)
    
    edge_subset = []
    # subgraph could switch edge labels, hence we have to iterate
    for e1 in list(G1.edges):
        if e1 in list(G.edges):
            edge_subset.append( list(G.edges).index(e1) )
        elif (e1[1], e1[0]) in list(G.edges):
            edge_subset.append(list(G.edges).index((e1[1], e1[0])))
            
    return G1, edge_subset
#%%
def plot_heatmap(D):
    # Define two rows for subplots
    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(15,6),  gridspec_kw={"height_ratios":[1,0.05]})
       
    sns.heatmap(D.T, ax = ax , square = False, cbar=False, xticklabels = D.index.values ,
                    linecolor = 'white',
                    cmap= sns.color_palette("YlGnBu", 100),
                    cbar_kws = dict(use_gridspec=False,location="bottom"))
    
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel('')
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
        
    cax.set_title('Color legend: standardized value of statistic', color = 'white')
    cax.tick_params(axis='x', colors='white')
    fig.set_facecolor("dimgrey")
    
    fig.subplots_adjust( left=0.14, right=0.95, hspace=0.7, wspace=0.2)
    
    return

def draw_network(G, names, clusters, edge_weights, write_labels = False, style = 'kamada'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    
    keys = list(G.nodes)
    sub_names = list(np.array(names)[keys])
    
    my_cmap = "viridis"
    label_dict = dict(zip(keys,sub_names))
    
    if  style == 'kamada':
        nx.draw_kamada_kawai(G, labels = label_dict, with_labels= write_labels,ax = ax, node_size = 70, node_color =  clusters,
                             cmap = my_cmap, vmin = -1, vmax = 1.5,
                             width = 2, edge_color = edge_weights, edge_cmap = plt.cm.binary_r, edge_vmin = 0.8, edge_vmax = 1.5, alpha = 1, font_color = 'white', font_size = 9)
    else :
        nx.draw_spring(G, labels = label_dict, with_labels= write_labels,ax = ax, node_size = 70, node_color =  clusters,
                             cmap = my_cmap, vmin = -1, vmax = 1.5,
                             width = 2, edge_color = edge_weights, edge_cmap = plt.cm.binary_r, edge_vmin = 0.8, edge_vmax = 1.5, alpha = 1, font_color = 'white', font_size = 9)
    
    fig.set_facecolor("dimgrey")
    
    #plt.savefig('res_network', facecolor=fig.get_facecolor(), transparent=True)