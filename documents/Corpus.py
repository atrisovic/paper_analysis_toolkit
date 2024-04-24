from documents.Paper import Paper, ReferenceSectionCountException
from citations.CitationClassifier import CitationClassifier
from citations.Reference import Reference
from citations.Agglomerator import RankedClassificationCounts, Agglomerator
from citations.FoundationModel import FoundationModel
from affiliations.AffiliationClassifier import AffiliationClassifier
from utils.functional import clusterOrLimitList, stemmed_basename

from typing import List, Tuple, Dict
from tqdm import tqdm
from os import walk
from os.path import join, basename

import pandas as pd, logging, json

logger = logging.getLogger(__name__)




class Corpus:
    def __init__(self, directory: str, 
                        extensions: List[str], 
                        foundation_model_limit: int = None, 
                        paper_limit: int = None,
                        cluster_info: Tuple[int, int] = None, 
                        filter_path: str = None,
                        lazy: bool = False,
                        confirm_paper_ref_sections: bool = True,
                        paper_years: Dict[str, int] = None):
        
        self.paper_limit: int = paper_limit
        self.foundation_model_limit: int = foundation_model_limit
        self.cluster_info: Tuple[int, int] = cluster_info
        self.directory: str = directory
        self.extensions: List[str] = extensions
        self.filter_path: str = filter_path
        self.lazy: bool = lazy
        self.confirm_paper_ref_sections: bool = confirm_paper_ref_sections
        self.paper_years = paper_years or {}
        
        self.setPapersLists()

    def setPapersLists(self) -> List[Paper]:
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

        if (self.paper_years):
            missing_ids = set(map(stemmed_basename, all_file_paths)) - set(self.paper_years.keys())
            assert(not missing_ids), f"Could not find years for {len(missing_ids)} ids. E.g.: {list(missing_ids)[0]}"
        
        good_papers, bad_papers = [], []
        for path in tqdm(all_file_paths):
            id = stemmed_basename(path)
            year = self.paper_years.get(id)
            try:
                good_papers.append(Paper(path, 
                                         lazy = self.lazy, 
                                         confirm_reference_section=self.confirm_paper_ref_sections,
                                         year = year)
                                   )
            except ReferenceSectionCountException as e:
                logger.debug(f"Exception occured creating Paper object from {path} (ignored, see Corpus.bad_papers) {e}")
                bad_papers.append((path, e))
                
        failure_rate = len(bad_papers)/(len(bad_papers) + len(good_papers))
        logger.info(f"Finished loading papers for corpus. {round(failure_rate * 100, 2)}% of papers threw an error. See Corpus.bad_papers.")
                        
        self.papers: List[Paper] = good_papers
        self.bad_papers: List[Tuple[str, Exception]] = bad_papers
 
        return good_papers, bad_papers
     
    def agglomerateResultsByTitle(self, id: str, agglomerator: Agglomerator = None):
        textRefByKey = [json for json in self.getAllTextualReferences(as_dict = True) if json.get('modelId') == id]
        df = pd.DataFrame.from_dict(textRefByKey)
        
        if (len(df) > 0):
            agglomerator = agglomerator or RankedClassificationCounts()            
            return agglomerator.applyQuery(df)

        return None
        
                

    def findAllPaperReferencesByTitle(self, model: FoundationModel, classifier: CitationClassifier):
        logger.info(f"Finding references for (key = {model.key}, title = {model.title[:30]}.). Classification is turned {'on' if classifier else 'off'}.")
        for paper in self.papers:
            paper.getReferenceFromTitle(model = model, classifier = classifier)
        logger.info(f"References successfully saved to underlying paper objects.")
            
    def findAllPaperRefsAllTitles(self, models = List[FoundationModel],
                                        classifier: CitationClassifier = None, 
                                        resultsfile = None,
                                        agglomerator: Agglomerator = None):
        
        models = clusterOrLimitList(models, self.cluster_info, self.foundation_model_limit)        

        f = open(resultsfile, 'a+') if resultsfile else None

        logger.info(f"Finding references to {len(models)} foundation models in corpus {'and' if classifier else 'without'} classifying sentences.")
        for model in tqdm(models):
            self.findAllPaperReferencesByTitle(model = model, classifier=classifier)
            
            results = self.agglomerateResultsByTitle(model.id, agglomerator = agglomerator) if f and classifier else None
            if results is not None:
                f.write(results.to_json(orient = 'index') + "\n")
                f.flush()
                logger.info(f"Saved all results for title = {model.title[:30]}... and key = {model.key}.")
            else:
                logger.info(f"No textual references found for {model.key}. Nothing written to results file.")

        if f:
            f.close()
                
    def agglomerateAllTextualReferences(self, agglomerator: Agglomerator):
        df = pd.DataFrame.from_dict(self.getAllTextualReferences(as_dict = True))
        classification_counts = agglomerator.applyQuery(df)
        return classification_counts
            
    def getAllReferences(self) -> List[Tuple[Paper, Reference]]:
        return [reference for paper in self.papers for ref_name, reference in paper.references.items()]
        
    def getAllTextualReferences(self, as_dict = False):
        return [row for paper in self.papers for row in paper.getAllTextualReferences(as_dict = as_dict)]
    
    def getAllAffiliations(self, classifier: AffiliationClassifier, resultsfile: str = None):
        f = open(resultsfile, 'a') if resultsfile else None
        for paper in tqdm(self.papers):
            logging.debug(f"Checking affiliation for paper at {paper.path}.")
            results = paper.getNamesAndAffiliations(classifier=classifier)
            
            if f:
                results_string = json.dumps({paper.path: results}) + '\n'
                logger.debug(f"Writing to {resultsfile}: {results_string}")
                f.write(results_string)
                f.flush()
                
        if f:
            f.close()