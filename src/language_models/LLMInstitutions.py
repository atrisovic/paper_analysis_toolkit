from pydantic import BaseModel as PydanticModel, conlist, constr
from src.language_models.FewShot import FewShotPipeline
import logging
from src.prompts.affiliation_prompts import PROMPT1
from src.language_models.ChatInterface import ChatInterface

logger = logging.getLogger(__name__)

    
class Institution(PydanticModel):
    name: constr(min_length = 1) # type: ignore
    type: constr(min_length = 1) #type:ignore #Literal['academic','industry','research']
    country: constr(min_length = 1) #type: ignore
    
    
class ListInstitutions(PydanticModel):
    institutions: conlist(item_type = Institution, min_length = 1) # type: ignore


class LLMInstitutions(FewShotPipeline):
    def __init__(self, 
                 interface: ChatInterface,
                 resultsfile: str = None, 
                 prompt = PROMPT1,
                 debug = False):
        
        #need to be careful with the brackets as we pass them around!
        self.interface = interface
        prompt = prompt.format(
                                schema = self.getSchema().replace('{', '{{').replace('}', '}}'), 
                                input = '{input}'
                               )
        
        super().__init__(interface = interface,
                        prompt = prompt,
                        debug = debug)

    def getExamples(self):
        return []
    
    def getSchema(self):
        paperAffiliations = self.interface.outputClass(institutions=[
                                        Institution(name = "Name of Institution1", type = "academic", country = 'Country1'),
                                        Institution(name = "Name of Institution2", type = "industry", country = 'Country2')
                                ])
        
        return paperAffiliations.model_dump_json(indent = 1)