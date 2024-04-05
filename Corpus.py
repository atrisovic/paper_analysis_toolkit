from typing import List, Tuple
from Paper import Paper
from math import ceil
from tqdm import tqdm
from CitationClassifier import CitationClassifier
from Reference import Reference
from os import walk, path
import pandas as pd
import logging

class Corpus:
    def __init__(self, directory, extensions, limit = 1000, cluster = None, cluster_count = None):
        self.limit = limit
        self.directory = directory
        self.extensions = extensions
        self.cluster = cluster
        self.cluster_count = cluster_count
        self.papers: List[Paper] = self.discoverPapers()

    def discoverPapers(self) -> List[Paper]:
        print(f"Reading all file in directory {self.directory}")
        all_file_paths = []
        for root, dirs, files in walk(self.directory):
            all_file_paths += [path.join(root, file)
                                    for file in files
                                        if (file.split('.')[-1] in self.extensions)]
        if (self.limit or self.cluster):
            all_file_paths = sorted(all_file_paths) # for consistency when clustering
            
        if (self.limit):
            all_file_paths = all_file_paths[:self.limit]
            
        if self.cluster:
            chunk_length = ceil(len(all_file_paths)/self.cluster_count)
            all_file_paths = all_file_paths[chunk_length * self.cluster: chunk_length * (self.cluster + 1)]
            
        self.papers = [Paper(path) for path in tqdm(all_file_paths)]
        return self.papers
    
        
    def saveClassificationByTitle(self, title: str, key: str, logfile: str):
        textRefByKey = [json for json in self.getAllTextualReferences(as_dict = True) if json.get('FM_key') == key]
        df = pd.DataFrame.from_dict(textRefByKey)
        
        if len(df) > 0:
          df['classification_ranking'] = df.groupby(['FM_key', 'paper'])['classification_order'].rank(method='min')
          classification_counts = (df[df['classification_ranking'] == 1]
                                                  .groupby(['FM_key', 'classification'])['paper']
                                                  .nunique()
                                                  .reset_index()
                                                  .rename(columns={'paper':'count'})
                                                  .pivot(index='FM_key', columns='classification', values='count')
                                                  .fillna(0)
                                                  .rename_axis(None, axis = 1)
                                              )
          with open(logfile, 'a+') as f:
              f.write(classification_counts.to_json(orient = 'index') + "\n")
            

    def findAllPaperReferencesByTitle(self, title: str, key: str, classifier: CitationClassifier):
        for paper in self.papers:
            paper.getReferenceFromTitle(title, key, classifier = classifier)
            
    def findAllPaperRefsAllTitles(self, titles: List[str], keys = List[str], classifier: CitationClassifier = None, logfile = None):
        print(f"Finding references to {len(titles)} titles in corpus {'and' if classifier else 'without'} classifying sentences.")
        for title, key in tqdm(list(zip(titles, keys))):
            self.findAllPaperReferencesByTitle(title = title, key = key, classifier=classifier)
            if logfile:
                self.saveClassificationByTitle(title, key, logfile)
            
            
    def getAllReferences(self) -> List[Tuple[Paper, Reference]]:
        return [reference for paper in self.papers for reference in paper.getAllReferences()]
        
    def getAllTextualReferences(self, as_dict = False):
        return [row for paper in self.papers for row in paper.getAllTextualReferences(as_dict = as_dict)]