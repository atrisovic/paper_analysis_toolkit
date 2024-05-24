import random
from math import ceil

class Cluster:
    def __init__(self, index = None, worker_count = None, limit = None, seed = 0):
        assert((worker_count is None) == (index is None)), "Precondition failed: worker_count and index should both be None or neither None."
        
        self.index = index or 1
        self.worker_count = worker_count or 1
        self.limit = limit #can be None
        self.seed = seed
    
    def clusterList(self, L: list):
        random.seed(self.seed)
        random.shuffle(L) #need consistency accross jobs
        
        L = L[:self.limit]
            
        chunk_length = ceil(len(L)/self.worker_count)
        
        L = L[chunk_length * (self.index - 1): chunk_length * self.index] #cluster_index is one-indexed

        return L