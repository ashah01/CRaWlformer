import numpy as np
import random
from networkx.algorithms.centrality import eigenvector_centrality_numpy
import torch

def centrality_nbrw(G, size):
    n = len(G.nodes())
    walks = np.empty((size, n), dtype=int)
    walk_edges = np.empty((size-1, n), dtype=int)
    for col in range(n):
        column(G, size, col, walks, walk_edges)
    return walks, walk_edges

def column(G, size, start, walks, walk_edges):
    """Finds sample from network G of the desired size using non-backtracking random walk 
        driven by node centrality. 
    
    Args: 
        G: networkX graph
        size: int desired size of the sample
        centrality_type: nb_centrality or built-in eigenvector_centrality_numpy
        dangling: str instruction on how to handle dangling nodes; 'backtrack' or 'remove'
    Returns: 
        S: list nodes in the sample
    """
    
    if size == 0:
        return []

    # initialise lists of sampled nodes and values
    S = np.zeros(size, dtype=int)
    E = np.zeros(size-1, dtype=int)

    # start at random node
    current = start
    ns = 1 
    S[0] = current
    centrality = eigenvector_centrality_numpy(G)
    while ns < size:
        
        if ns == 1:
            neighbours = list(G.neighbors(current))
            neighbour_centralities = np.array([centrality[x] for x in neighbours])
            # normalise probability 
            prob = neighbour_centralities / sum(neighbour_centralities)
            untraveled_mask = ~np.isin(neighbours, S)
            if untraveled_mask.any():
                node = neighbours[np.where(prob==prob[untraveled_mask].max())[0][0]]
            else:
                # choose the neighbour with the highest centrality    
                node = np.random.choice(neighbours)
        else: 
            neighbours = list(G.neighbors(current))
            if len(neighbours) == 1:
                # make walker backtrack if it reaches dangling node
                node = S[ns-2]
            else: 
                # remove previous node from list of neighbours
                neighbours.remove(S[ns-2])
                
                # compute degree centrality for neighbouring nodes
                neighbour_centralities = np.array([centrality[x] for x in neighbours])
                prob = neighbour_centralities / sum(neighbour_centralities)
                untraveled_mask = ~np.isin(neighbours, S)
                if untraveled_mask.any():
                    node = neighbours[np.where(prob==prob[untraveled_mask].max())[0][0]]
                else:
                    # choose the neighbour with the highest centrality    
                    node = np.random.choice(neighbours)
        
        i = 0
        for edge in G.edges:
            if edge == (current, node):
                break
            i += 1
        E[ns-1] = i
        S[ns] = node
        current = node
        ns += 1
        
    walks[:, start] = (S.astype(int))
    walk_edges[:, start] = (E.astype(int))