import jsonlines
from pydantic import BaseModel
from typing import List
from json import dumps
from src.document.paper import Paper
from src.process.FoundationModel import FoundationModel

CITATION_GRAPH_PATH = '/home/gridsan/afogelson/osfm/paper_analysis_toolkit/scripts/missing_fm_debugging/citation_graph_missing_mk_present_0927.jsonl'

class MissingFoundationModel(BaseModel):
    paperId: str
    citingPapers: List[str]
    modelKey: str
    title: str
    
class MissingPair(BaseModel):
    modelId: str
    paperId: str
    modelKey: str
    title: str 
    
def get_missing_pairs():
    pairs = []
    with jsonlines.open(CITATION_GRAPH_PATH) as reader:
        for obj in reader:
            fm = MissingFoundationModel.model_validate_json(dumps(obj))
            pairs += [MissingPair(modelId = fm.paperId, paperId = paperId, modelKey = fm.modelKey, title = fm.title) for paperId in fm.citingPapers]
    return pairs



def print_references():
    pairs = get_missing_pairs()
    for missing_pair in pairs:
        fm = FoundationModel(key = missing_pair.modelKey, title= missing_pair.title, id= missing_pair.modelId, year=None)
        sus_paper = Paper(f'../data/markdown/{missing_pair.paperId}.mmd')
        ref = sus_paper.getReferenceFromTitle(fm)
        print(sus_paper.references[fm.key])
        