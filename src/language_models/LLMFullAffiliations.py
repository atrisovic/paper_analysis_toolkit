from pydantic import BaseModel as PydanticModel, conlist, model_validator
from src.language_models.FewShot import FewShotPipeline
import logging
from src.prompts.affiliation_prompts import PROMPT1
from typing import List, Literal
from src.language_models.ChatInterface import HFChatInterface


logger = logging.getLogger(__name__)


class Contributor(PydanticModel):
    first: str
    last: str
    gender: Literal['male', 'female']
    
    
class Institution(PydanticModel):
    name: str
    type: Literal['academic', 'industry']
    
    
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


class LLMFullAffiliationsPipepline(FewShotPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 device, 
                 resultsfile: str = None, 
                 prompt = PROMPT1,
                 debug =  False,
                 strict = True):
        
        #need to be careful with the brackets as we pass them around!
        prompt = prompt.format(
                                schema = self.getSchema().replace('{', '{{').replace('}', '}}'), 
                                input = '{input}'
                               )
        interface = HFChatInterface(model, 
                                    tokenizer, 
                                    device, 
                                    PaperAffiliationsStrict if strict else PaperAffiliationsNonStrict, 
                                    debug = debug,
                                    resultsfile = resultsfile)

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