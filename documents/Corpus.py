from typing import List, Tuple
from .Paper import Paper
from tqdm import tqdm
from citations.CitationClassifier import CitationClassifier
from citations.Reference import Reference
from affiliations.AffiliationClassifier import AffiliationClassifier
from os import walk
from os.path import join, basename
import pandas as pd
import logging
from utils.functional import clusterOrLimitList
import json

logger = logging.getLogger(__name__)

class Corpus:
    def __init__(self, directory, 
                        extensions: List[str], 
                        foundation_model_limit: int = None, 
                        paper_limit: int = None,
                        cluster_info: Tuple[int, int] = None, 
                        filter_path: str = None,
                        lazy: bool = False):
        
        self.paper_limit = paper_limit
        self.foundation_model_limit = foundation_model_limit
        self.cluster_info = cluster_info
        self.directory = directory
        self.extensions = extensions
        self.filter_path = filter_path
        self.lazy = lazy
        
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
            
        if (self.filter_path):
            with open(self.filter_path, 'r') as f:
                filtered_content = f.read().lower()
            all_file_paths = list(filter(lambda s: basename(s).split('.')[0] in filtered_content, all_file_paths))
        
        logger.info(f"Found {len(all_file_paths)} files (filter_path set to {self.filter_path}).")
            
            
        all_file_paths = clusterOrLimitList(all_file_paths, self.cluster_info, self.paper_limit)
        
        logger.info(f"Loading {len(all_file_paths)} files as Paper objects." )
        good_papers, bad_papers = [], []
        for path in tqdm(all_file_paths):
            try:
                good_papers.append(Paper(path, lazy = self.lazy)) #technically the append could fail, keep this in mind
            except AssertionError as e:
                logger.debug(f"Exception occured creating Paper object from {path} (ignored, see Corpus.bad_papers) {e}")
                bad_papers.append((path, e))
        failure_rate = len(bad_papers)/(len(bad_papers) + len(good_papers))
        logger.info(f"Finished loading papers for corpus. {round(failure_rate * 100, 2)}% of papers threw an error. See Corpus.bad_papers.")
                        
 
        return good_papers, bad_papers
    
        
    def saveClassificationByTitle(self, title: str, key: str, resultsfile: str):
        logger.info(f"Saving group classification metrics to {resultsfile} for (key={key}, title={title[:30]}.).")
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
        logger.info(f"Finding references for (key = {key}, title = {title[:30]}.). Classification is turned {'on' if classifier else 'off'}.")
        for paper in self.papers:
            paper.getReferenceFromTitle(title, key, classifier = classifier)
        logger.info(f"References successfully saved to underlying paper objects.")
            
    def findAllPaperRefsAllTitles(self, titles: List[str], keys = List[str], classifier: CitationClassifier = None, resultsfile = None):
        titles = clusterOrLimitList(titles, self.cluster_info, self.foundation_model_limit)
        keys = clusterOrLimitList(keys, self.cluster_info, self.foundation_model_limit)

        logger.info(f"Finding references to {len(titles)} titles in corpus {'and' if classifier else 'without'} classifying sentences.")
        for title, key in tqdm(list(zip(titles, keys))):
            self.findAllPaperReferencesByTitle(title = title, key = key, classifier=classifier)
            if resultsfile and classifier:
                self.saveClassificationByTitle(title, key, resultsfile)
            
            
    def getAllReferences(self) -> List[Tuple[Paper, Reference]]:
        return [reference for paper in self.papers for reference in paper.getAllReferences()]
        
    def getAllTextualReferences(self, as_dict = False):
        return [row for paper in self.papers for row in paper.getAllTextualReferences(as_dict = as_dict)]
    
    def setAllAffiliations(self, classifier: AffiliationClassifier, resultsfile: str = None):
        f = open(resultsfile, 'a') if resultsfile else None
        for paper in tqdm(self.papers):
            logging.debug(f"Checking affiliation for paper at {paper.path}.")
            results = paper.findNamesAndAffiliations(classifier=classifier)
            if f:
                results_string = json.dumps({paper.path: results}) + '\n'
                logger.debug(f"Writing to {resultsfile}: {results_string}")
                f.write(results_string)
        f.close()