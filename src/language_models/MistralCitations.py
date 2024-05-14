from typing import Literal
from src.language_models.FewShot import FewShotPipeline
from pydantic import BaseModel as PydanticModel
from src.language_models.OutputParser import OutputParser

    
class Classification(PydanticModel):
    classification: Literal['uses', 'extends', 'background', 'motivation', 'future_work', 'differences']
    
class MistralCitationPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, device = None):
        super().__init__(model = model, tokenizer=tokenizer, device=device)
    
        self.outputParser = OutputParser(outputClass = Classification)
    def wrapInstructions(self, input: str):
        return f"""The following sentence is from an academic paper which cites a foundation model (a pretrained machine learning model for a particular task). 
The particular model isn't important, but we'd like to discern *how* the paper makes use of the model. In particular, we want to classify as 'uses' if the sentence 
indicates that the paper only deploys the model without modifications, 'extends' if it suggests specific alterations to the model that change its structure or 
functioning, 'background' for general context about the area, 'motivation' for reasons behind the research, 'future_work' for proposed next steps in the research, 
and 'differences' for comparisons with other work. Please response in JSON format 
{{'classification': 'uses | extends | background | motivation | future_work | differences'}}, classifying the following sentence {input}"""   

    def generateAsModel(self, input: str, tolerance= 5, paperId: str = None) -> PydanticModel:
        counter = 0
        output_object = None
        
        while counter < tolerance and not output_object:
            counter += 1
            results = self.generate(input = input)
            output_object = self.outputParser.parse(results)
             
        
        return output_object