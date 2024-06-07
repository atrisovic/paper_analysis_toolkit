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
        L.sort()
        random.shuffle(L) #need consistency accross jobs
            
        return L[self.index - 1: self.limit : self.worker_count] #cluster_index is one-indexed