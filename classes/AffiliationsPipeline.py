from torch import TensorType
import regex as re
import json
from os.path import join
from typing import List
from tqdm import tqdm

from pydantic import BaseModel as PydanticModel, ValidationError
import re
from typing import Dict, List, Literal, Tuple
from classes.FewShot import FewShotPipeline, PaperAffiliations, Contributor, Institution
from transformers import Pipeline



class AffiliationsPipeline(FewShotPipeline):
    def __init__(self, pipeline: Pipeline, outputClass: PydanticModel = PaperAffiliations, resultsfile: str = None):
        super().__init__(pipeline = pipeline, outputClass=outputClass)
        
        for question, answer in self.getExamples():
            self.addExample(question=question, answer=answer)


    def getExamples(self):
        example_text = "# on the helpfulness of large language models\n\nbill ackendorf, Jolene baylor\n\nyuxin shu, khalid saifullah\n\nalex fogelson ({}^{\\dagger}), ana trisovic, neil thompson({}^{\\ddagger}), bob dilan\n\n({}^{\\ddagger}) new york university, massachusetts institute of technology"

        paperAffiliations = PaperAffiliations(
                        contributors = [ 
                                        Contributor(first = "Bill", last= "Ackendorf", gender= "male"),
                                        Contributor(first= "Jolene", last= "Baylor", gender= "female"),
                                        Contributor(first= "Yuxin", last= "Shu", gender= "female"),
                                        Contributor(first= "Alex", last= "Fogelson", gender= "male"),
                                        Contributor(first= "Ana", last= "Trisovic", gender= "female"),
                                        Contributor(first= "Neil", last= "Thompson", gender= "male"),
                                ],
                        institutions = [
                                        Institution(name = "New York University", type = "academic"),
                                        Institution(name = "Massachusetts Institute of Technology", type =  "academic")
                                ],
                        countries = ["United States"]
        )
        
        return {example_text: paperAffiliations}.items()
    

    def classifyFromTextEnsureJSON(self, text: str, tolerance=5) -> dict:
        counter = 0
        json_result = None
        while counter < tolerance and not json_result:
            counter += 1
            json_string_results = self.classifyFromText(text)
            try:
                json_result = json.loads(json_string_results)
            except:
                pass

        try:
            structured_json = (
                self.stripJSONStructure(json_result) if json_result else None
            )
        except:
            structured_json = None

        return structured_json

