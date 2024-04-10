from typing import List, Tuple
from Paper import Paper
from math import ceil
from tqdm import tqdm
from CitationClassifier import CitationClassifier
from Reference import Reference
from os import walk
from os.path import join
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Corpus:
    def __init__(self, directory, extensions, limit = 1000, cluster = None, cluster_count = None):
        self.limit = limit
        self.directory = directory
        self.extensions = extensions
        self.cluster = cluster
        self.cluster_count = cluster_count
        
        good_papers, bad_papers = self.discoverPapers()
        self.bad_papers: List[Tuple[str, Exception]] = bad_papers
        self.papers: List[Paper] = good_papers

    def discoverPapers(self) -> List[Paper]:
        logger.info(f"Discovering all files in directory {self.directory} with extensions in {self.extensions}.")
        all_file_paths = []
        for root, dirs, files in walk(self.directory):
            all_file_paths += [join(root, file)
                                    for file in files
                                        if (file.split('.')[-1] in self.extensions)]
        
        logger.info(f"Found {len(all_file_paths)} files. Limiting based on \"limit\" and \"cluster_count\"")
        if (self.limit or self.cluster):
            all_file_paths = sorted(all_file_paths) # for consistency when clustering
            
        if (self.limit):
            all_file_paths = all_file_paths[:self.limit]
            
        if self.cluster:
            chunk_length = ceil(len(all_file_paths)/self.cluster_count)
            all_file_paths = all_file_paths[chunk_length * self.cluster: chunk_length * (self.cluster + 1)]
            
        logger.info(f"Loading {len(all_file_paths)} files as Paper objects." )
        good_papers, bad_papers = [], []
        for path in tqdm(all_file_paths):
            try:
                good_papers.append(Paper(path)) #technically the append could fail, keep this in mind
            except Exception as e:
                logger.debug(f"Exception occured creating Paper object from {path} (ignored, see Corpus.bad_papers) {e}")
                bad_papers.append((path, e))
        failure_rate = len(bad_papers)/(len(bad_papers) + len(good_papers))
        logger.info(f"Finished loading papers for corpus. {round(failure_rate * 100, 2)}% of papers threw an error. See Corpus.bad_papers.")
                        
 
        return good_papers, bad_papers
    
        
    def saveClassificationByTitle(self, title: str, key: str, resultsfile: str):
        logger.info(f"Saving group classification metrics to {resultsfile} for (key={key}, title={title[:30]}...).")
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
            with open(resultsfile, 'a+') as f:
                f.write(classification_counts.to_json(orient = 'index') + "\n")
            logger.info(f"Successfully computed and saved results.")
        else:
            logger.info(f"No textual references found for {key}. Nothing written to results file.")

    def findAllPaperReferencesByTitle(self, title: str, key: str, classifier: CitationClassifier):
        logger.info(f"Finding references for (key = {key}, title = {title[:30]}...). Classification is turned {'on' if classifier else 'off'}.")
        for paper in self.papers:
            paper.getReferenceFromTitle(title, key, classifier = classifier)
        logger.info(f"References successfully saved to underlying paper objects.")
            
    def findAllPaperRefsAllTitles(self, titles: List[str], keys = List[str], classifier: CitationClassifier = None, resultsfile = None):
        logger.info(f"Finding references to {len(titles)} titles in corpus {'and' if classifier else 'without'} classifying sentences.")
        for title, key in tqdm(list(zip(titles, keys))):
            self.findAllPaperReferencesByTitle(title = title, key = key, classifier=classifier)
            if resultsfile and classifier:
                self.saveClassificationByTitle(title, key, resultsfile)
            
            
    def getAllReferences(self) -> List[Tuple[Paper, Reference]]:
        return [reference for paper in self.papers for reference in paper.getAllReferences()]
        
    def getAllTextualReferences(self, as_dict = False):
        return [row for paper in self.papers for row in paper.getAllTextualReferences(as_dict = as_dict)]