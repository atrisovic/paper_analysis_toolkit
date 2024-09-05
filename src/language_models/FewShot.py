from pydantic import BaseModel as PydanticModel
from typing import Dict, List, Union
import json
from src.language_models.OutputParser import OutputParser
import logging
from torch.cuda import OutOfMemoryError
import os
from src.language_models.ChatInterface import ChatInterface

logger = logging.getLogger(__name__)
username = os.getenv('USER') or os.getenv('USERNAME')

class FewShotExample(PydanticModel):
    input: str
    output: str
    
    
class FewShotPipeline:
    examples: List[Dict]
    
    def __init__(self, 
                 interface: ChatInterface,
                 prompt: str, 
                 outputClass: PydanticModel = None, 
                 resultsfile: str = None,
                 debug: bool = True):
        
        self.interface = interface
        self.examples = []
        self.outputParser = OutputParser(outputClass = outputClass, 
                                         logfile = f'/home/gridsan/{username}/osfm/paper_analysis_toolkit/temp.log')
        self.resultsfile = resultsfile
        self.prompt = prompt
        self.debug = debug
        
        for question, answer in self.getExamples():
            self.addExample(question=question, answer=answer)
    
    # Implemented in children
    def getExamples(self):
        return []
    
    def wrapInstructions(self, input: str):
        return self.prompt.format(input = input)   

    # Save few-shot example, either as string or existing base model.
    def addExample(self, question: str, answer: Union[str, PydanticModel]):
        if (isinstance(answer, PydanticModel)):
            example = FewShotExample(input = question, output = answer.model_dump_json(indent=1))
        else:
            example = FewShotExample(input = question, output = answer)
            
        self.examples.append(example)   
    
    # Generate list of messages for LLM input prompt, without few shot context
    def generateZeroShotPrompt(self, input: str, output: str = None) -> List[Dict]:
        zero_shot = [{"role": "user", "content": self.wrapInstructions(input)}]
        
        if output:
            zero_shot.append({"role": "assistant", "content": output})
        
        return zero_shot

    # Merge multiple zero shot prompts to create few shot prompt based on input and pre-added examples
    def getFewShotPrompt(self, input: str, max_examples: int = None):
        few_shots = []
        for example in self.examples[:max_examples]:
            few_shots += self.generateZeroShotPrompt(example.input, example.output)
            
        few_shots += self.generateZeroShotPrompt(input = input, output = None)
            
        return few_shots

    def generate(self, input: str, 
                        max_examples: int = None, 
                        strict = False, 
                        tolerance = 5, 
                        identifier: str = None, 
                        strip_output: bool = True, 
                        last_attempt = False):   
            
        if (self.examples):
            input = self.getFewShotPrompt(input, max_examples = max_examples)
        
        if (strict):
            return self.interface.generateAsModel(input = input, 
                                        tolerance=tolerance, 
                                        identifier=identifier, 
                                        strip_output = strip_output, 
                                        last_attempt=last_attempt)
        else:
            return self.interface.generate(input, temperature=.5)
    

    
    