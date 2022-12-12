import numpy as np
import random
from networkx.algorithms.centrality import eigenvector_centrality_numpy
import torch

def centrality_nbrw(G, size, dangling=None):
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
    
    if dangling == 'remove':
        remove = [node for node,degree in dict(G.degree()).items() if degree == 1]
        G.remove_nodes_from(remove)

    # initialise lists of sampled nodes and values
    S = np.zeros(size)
    E = np.zeros(size-1)
    n = len(G.nodes())
    y = np.zeros(size)

    # start at random node
    current = random.choice(list(G.nodes()))
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
                if dangling == 'backtrack':
                    node = S[ns-2]
                else: 
                    raise Exception('The walker got stuck at an absorbing node.')    
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
        
    return torch.from_numpy(S.astype(int)), torch.from_numpy(E.astype(int))