from pydantic import BaseModel as PydanticModel, conlist
from src.language_models.FewShot import FewShotPipeline
import logging
from src.prompts.affiliation_prompts import PROMPT1
from typing import List, Literal
from .OutputParser import OutputParser

logger = logging.getLogger(__name__)


class Contributor(PydanticModel):
    first: str
    last: str
    gender: Literal['male', 'female']
    
    
class Institution(PydanticModel):
    name: str
    type: Literal['academic', 'industry']
    
    
class PaperAffiliations(PydanticModel):
    contributors: conlist(item_type = Contributor, min_length = 1) # type: ignore
    institutions: conlist(item_type = Institution, min_length = 1) # type: ignore
    countries: conlist(item_type = str, min_length = 1) #type: ignore


class LLMAffiliationsPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, device, resultsfile: str = None, prompt = PROMPT1):
        
        #need to be careful with the brackets as we pass them around!
        prompt = prompt.format(
                                schema = self.getSchema().replace('{', '{{').replace('}', '}}'), 
                                input = '{input}'
                               )
        super().__init__(model = model, 
                        tokenizer=tokenizer, 
                        device = device, 
                        outputClass = PaperAffiliations, 
                        resultsfile=resultsfile,
                        prompt = prompt)

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
    
    def getSchema(self):
        paperAffiliations = PaperAffiliations(
                        contributors = [ 
                                        Contributor(first = "firstname", last= "lastname", gender= "male"),
                                        Contributor(first= "firstname", last= "lastname", gender= "female")
                                ],
                        institutions = [
                                        Institution(name = "University of City", type = "academic"),
                                        Institution(name = "Major Company", type =  "industry")
                                ],
                        countries = ["United States", "China", "Other Countries"]
        )
        
        return paperAffiliations.model_dump_json(indent = 1)