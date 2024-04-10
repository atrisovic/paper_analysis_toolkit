import pandas as pd
from typing import Tuple
from math import ceil
from torch import backends, cuda

df = pd.read_csv('./data/sub_sample_dataset2.csv')
id_to_title = {k:v for k, v in zip(df['citation.paperId'].to_list(), df['citation.title'].to_list())} | {k:v for k, v in zip(df['model.paperId'].to_list(), df['model.title'].to_list())}


def implies(a: bool, b: bool) -> bool:
    return not(a) or b

def title_from_path(path:str):
        id = path.split('/')[-1].replace('.mmd','')
        return id_to_title.get(id)
    
def clusterOrLimitList(L: list, cluster_info: Tuple[int, int] = None, limit: int = None):
    if (not limit and not cluster_info):
        return L

    L = sorted(L) #need consistency accross jobs
    
    L = L[:limit or len(L)]
        
    if cluster_info:
        cluster_index, cluster_count = cluster_info
        chunk_length = ceil(len(L)/cluster_count)
    
        L = L[chunk_length * (cluster_index - 1): chunk_length * cluster_index] #cluster_index is one-indexed

    return L

