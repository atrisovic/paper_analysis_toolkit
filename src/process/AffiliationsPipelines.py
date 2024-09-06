from pydantic import BaseModel as PydanticModel, conlist, constr, model_validator
from src.language_models.FewShot import FewShotPipeline
import logging
from src.prompts.affiliation_prompts import PROMPT1
from src.language_models.ChatInterface import ChatInterface, HFChatInterface
from typing import List, Literal


logger = logging.getLogger(__name__)


##### OUTPUT VERSION 1 #####
    
class Institution(PydanticModel):
    name: constr(min_length = 1) # type: ignore
    type: constr(min_length = 1) #type:ignore #Literal['academic','industry','research']
    country: constr(min_length = 1) #type: ignore
    
    
class ListInstitutions(PydanticModel):
    institutions: conlist(item_type = Institution, min_length = 1) # type: ignore

##### OUTPUT VERSION 2 & 3 #####

class Contributor(PydanticModel):
    first: str
    last: str
    gender: Literal['male', 'female']
    
class PaperAffiliationsStrict(PydanticModel):
    contributors: conlist(item_type = Contributor, min_length = 1) # type: ignore
    institutions: conlist(item_type = Institution, min_length = 1) # type: ignore
    countries: conlist(item_type = str, min_length = 1) #type: ignore
    
class PaperAffiliationsNonStrict(PydanticModel):
    contributors: List[Contributor]
    institutions: List[Institution]
    countries: List[str]
    
    @model_validator(mode='after')
    def not_all_empty(self):
        assert(len(self.institutions) != 0 or len(self.countries) == 0)
        assert(self.contributors or self.institutions or self.countries)


##### ZERO SHOT #####

class ZeroShotAffiliationsPipeline(FewShotPipeline):
    def __init__(self, 
                 interface: ChatInterface,
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
    
##### FEW SHOT #####

class FewShotAffiliationsPipeline(FewShotPipeline):
    def __init__(self, 
                 interface: ChatInterface,
                 prompt = PROMPT1,
                 debug =  False)
        
        #need to be careful with the brackets as we pass them around!
        self.interface = interface
        prompt = prompt.format(
                                schema = self.getSchema().replace('{', '{{').replace('}', '}}'), 
                                input = '{input}'
                               )
        

        super().__init__(interface=interface,
                        prompt = prompt,
                        debug = debug)

    def getExamples(self):
        example_text = "# on the helpfulness of large language models\n\nbill ackendorf, Jolene baylor\n\nyuxin shu, khalid saifullah\n\nalex fogelson ({}^{\\dagger}), ana trisovic, neil thompson({}^{\\ddagger}), bob dilan\n\n({}^{\\ddagger}) new york university, massachusetts institute of technology"

        paperAffiliations = PaperAffiliationsStrict(
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
        paperAffiliations = PaperAffiliationsStrict(
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