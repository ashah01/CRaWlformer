import numpy as np
from networkx.algorithms.centrality import eigenvector_centrality_numpy

class RandomWalk:
    def __init__(self, G, size):
        self.G = G
        self.size = size
        self.centrality = eigenvector_centrality_numpy(G)
        self.n = len(G.nodes())
        self.current = np.array(G.nodes())
        self.walks = np.zeros((size, self.n), dtype=int)
        self.walk_edges = np.zeros((size-1, self.n), dtype=int)
        self.walks[0] = self.current
    
    def walk(self):
        for step in range(self.size - 1):
            self.row(step + 1)

        return self.walks, self.walk_edges

    def row(self, steps):
        neighbours = [list(self.G.neighbors(node)) for node in self.current]
        node = np.zeros_like(self.G.nodes())
        edges = np.zeros_like(self.G.nodes())
        if steps == 1:
            
            neighbour_centralities = [np.array([self.centrality[j] for j in x]) for x in neighbours]
            # normalise probability 
            prob = [neighbourhood / sum(neighbourhood) for neighbourhood in neighbour_centralities]
            untraveled_mask = [~np.isin(neighbours[index], self.walks[:, index]) for index in range(self.n)]
            for i, mask in enumerate(untraveled_mask):
                if mask.any():
                    node[i] = neighbours[i][np.where(prob[i] == prob[i][mask].max())[0][0]]
                else:
                    node[i] = np.random.choice(neighbours[i])
        else:
            for i, neighbourhood in enumerate(neighbours):
                if len(neighbourhood) == 1:
                    node[i] = self.walks[:, i][steps-2]
                else: 
                    # remove previous node from list of neighbours
                    neighbourhood.remove(self.walks[:, i][steps-2])
                    
                    # compute degree centrality for neighbouring nodes
                    neighbour_centralities = np.array([self.centrality[x] for x in neighbourhood])
                    prob = neighbour_centralities / sum(neighbour_centralities)
                    untraveled_mask = ~np.isin(neighbourhood, self.walks[:, i])
                    if untraveled_mask.any():
                        node[i] = neighbourhood[np.where(prob==prob[untraveled_mask].max())[0][0]]
                    else:
                        # choose the neighbour with the highest centrality    
                        node[i] = np.random.choice(neighbourhood)
        
        self.walks[steps] = node
        for index, edge in enumerate(self.G.edges):
            for i in range(self.n):
                if edge == (self.current[i], node[i]):
                    edges[i] = index
            
        self.walk_edges[steps-1] = edges
        self.current = node