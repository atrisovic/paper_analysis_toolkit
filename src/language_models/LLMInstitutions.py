from pydantic import BaseModel as PydanticModel, conlist, constr
from src.language_models.FewShot import FewShotPipeline
import logging
from src.prompts.affiliation_prompts import PROMPT1
from typing import List, Literal

logger = logging.getLogger(__name__)

    
class Institution(PydanticModel):
    name: constr(min_length = 1) # type: ignore
    type: Literal['academic','industry','research']
    country: constr(min_length = 1) #type: ignore
    
    
class ListInstitutions(PydanticModel):
    institutions: conlist(item_type = Institution, min_length = 1) # type: ignore


class LLMInstitutions(FewShotPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 device, 
                 resultsfile: str = None, 
                 prompt = PROMPT1,
                 debug = False):
        
        self.outputClass = ListInstitutions
        #need to be careful with the brackets as we pass them around!
        prompt = prompt.format(
                                schema = self.getSchema().replace('{', '{{').replace('}', '}}'), 
                                input = '{input}'
                               )
        super().__init__(model = model, 
                        tokenizer=tokenizer, 
                        device = device, 
                        outputClass = ListInstitutions, 
                        resultsfile=resultsfile,
                        prompt = prompt,
                        debug = debug)

    def getExamples(self):
        return []
        example_text = "# on the helpfulness of large language models\n\nbill ackendorf, Jolene baylor\n\nyuxin shu, khalid saifullah\n\nalex fogelson ({}^{\\dagger}), ana trisovic, neil thompson({}^{\\ddagger}), bob dilan\n\n({}^{\\ddagger}) new york university, massachusetts institute of technology"

        paperInstitutions = self.outputClass(institutions=[
                                        Institution(name = "New York University", type = "academic", country = 'United States'),
                                        Institution(name = "Massachusetts Institute of Technology", type = "academic", country = 'United States')
                                ]
        )
        
        return {example_text: paperInstitutions}.items()
    
    def getSchema(self):
        paperAffiliations = self.outputClass(institutions=[
                                        Institution(name = "Name of Institution1", type = "academic", country = 'Country1'),
                                        Institution(name = "Name of Institution2", type = "industry", country = 'Country2')
                                ])
        
        return paperAffiliations.model_dump_json(indent = 1)