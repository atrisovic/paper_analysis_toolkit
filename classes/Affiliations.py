from pydantic import BaseModel as PydanticModel
from classes.FewShot import FewShotPipeline
import logging, json
from typing import List, Literal
from .Parser import OutputParser

logger = logging.getLogger(__name__)


class Contributor(PydanticModel):
    first: str
    last: str
    gender: Literal['male', 'female']
    
    
class Institution(PydanticModel):
    name: str
    type: Literal['academic', 'industry']
    
    
class PaperAffiliations(PydanticModel):
    contributors: List[Contributor]
    institutions: List[Institution]
    countries: List[str]


class AffiliationsPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, device, 
                        resultsfile: str = None):
        super().__init__(model = model, tokenizer=tokenizer, device = device)
        
        self.outputParser = OutputParser(outputClass = PaperAffiliations) 
        self.resultsfile = resultsfile
        
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
    
    def wrapInstructions(self, input: str):
        return f"""Please read this text, and return the following information in the JSON format provided: 
{self.getSchema()}\n
The output should match exactly the JSON format below. The text is as follows:\n\n {input}"""      
    

    def generateAsModel(self, input: str, tolerance= 5, paperId: str = None) -> PydanticModel:
        counter = 0
        output_object = None
        
        while counter < tolerance and not output_object:
            counter += 1
            results = self.generate(input = input)
            output_object = self.outputParser.parse(results)
            

        if (self.resultsfile and output_object):
            assert(id is not None), f"Found resultsfile but paperId parameter not passed."
            json_result = {paperId: output_object.model_dump()}
            with open(self.resultsfile, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        else:
            json_result = {paperId: None}
            with open(self.resultsfile, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        
        return output_object


