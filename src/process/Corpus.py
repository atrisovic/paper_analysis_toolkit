from src.document.paper import Paper, ReferenceSectionCountException
from src.classifier.CitationClassifier import CitationClassifier
from src.document.reference import Reference
from src.process.FoundationModel import FoundationModel
from src.language_models.LLMFullAffiliations import LLMFullAffiliationsPipepline
from src.functional import stemmed_basename
from src.process.Cluster import Cluster

from gc import collect

from typing import List, Tuple, Dict
from tqdm import tqdm
from os import walk
from os.path import join, basename


import pandas as pd, logging



logger = logging.getLogger(__name__)


class Corpus:
    def __init__(self, directory: str, 
                        extensions: List[str], 
                        cluster: Cluster,
                        cluster_foundation_models: bool = False,
                        filter_path: str = None,
                        lazy: bool = False,
                        confirm_paper_ref_sections: bool = True,
                        paper_years: Dict[str, int] = None):
        
        self.cluster: Cluster = cluster
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
            
            
        all_file_paths = self.cluster.clusterList(all_file_paths)
        
        logger.info(f"Loading {len(all_file_paths)} files as Paper objects." )

        if (self.paper_years):
            missing_ids = set(map(stemmed_basename, all_file_paths)) - set(self.paper_years.keys())
            #assert(not missing_ids), f"Could not find years for {len(missing_ids)} ids. E.g.: {list(missing_ids)[0]}"
        
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
  
    def findAllReferencesForModel(self, model: FoundationModel, classifier: CitationClassifier):
        logger.info(f"Finding references for (key = {model.key}, title = {model.title[:30]}.). Classification is turned {'on' if classifier else 'off'}.")
        for paper in self.papers:
            paper.getReferenceFromTitle(model = model, classifier = classifier)
        logger.info(f"References successfully saved to underlying paper objects.")
            
    def findAllReferencesAllModels(self, models = List[FoundationModel],
                                        classifier: CitationClassifier = None,
                                        resultsfile: str = None):

        logger.info(f"Finding references to {len(models)} foundation models in corpus {'and' if classifier else 'without'} classifying sentences.")
        header = True
        for model in tqdm(models):
            self.findAllReferencesForModel(model = model, classifier=classifier)
            
            textRefByKey = [json for json in self.getAllTextualReferences(as_dict = True) if json.get('modelId') == model.id]
            df = pd.DataFrame.from_dict(textRefByKey)
            if resultsfile and len(df) > 0:
                df.to_csv(resultsfile, mode='a', index=False, header=header)
                header = False
    
            
    def getAllReferences(self) -> List[Tuple[Paper, Reference]]:
        return [reference for paper in self.papers for _, reference in paper.references.items()]
        
    def getAllTextualReferences(self, as_dict = False):
        return [row for paper in self.papers for row in paper.getAllTextualReferences(as_dict = as_dict)]
    
    def getAllAffiliations(self, pipeline: LLMFullAffiliationsPipepline):
        for paper in tqdm(self.papers):
            logging.debug(f"Checking affiliation for paper at {paper.path}.")
            paper.getNamesAndAffiliations(pipeline=pipeline)
            collect()