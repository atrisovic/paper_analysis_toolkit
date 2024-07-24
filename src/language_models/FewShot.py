from pydantic import BaseModel as PydanticModel
from typing import Dict, List, Union
import json
from src.language_models.OutputParser import OutputParser
import logging
from torch.cuda import OutOfMemoryError

logger = logging.getLogger(__name__)


class FewShotExample(PydanticModel):
    input: str
    output: str
    
    
    
class FewShotPipeline:
    examples: List[Dict]
    
    def __init__(self, 
                 model, 
                 tokenizer, 
                 prompt: str, 
                 device = None, 
                 outputClass: PydanticModel = None, 
                 resultsfile: str = None,
                 debug: bool = True):
        
        self.model = model
        self.tokenizer = tokenizer
        self.examples = []
        self.device = device
        self.outputClass = outputClass
        self.outputParser = OutputParser(outputClass = outputClass, 
                                         logfile = '/home/gridsan/afogelson/osfm/paper_analysis_toolkit/temp.log')
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
    
    # Generate using few shot prompt
    def generate(self, input: str, max_examples: int = None):        
        few_shot = self.getFewShotPrompt(input, max_examples = max_examples)
        
        encodeds = self.tokenizer.apply_chat_template(few_shot, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, 
                                            max_new_tokens=5000, 
                                            pad_token_id = self.tokenizer.eos_token_id, 
                                            do_sample=True, 
                                            temperature=.5)
        decoded = self.tokenizer.batch_decode(generated_ids)[0]
        
            
        return decoded
    
    
    # Iteratively generate output and force output to match a Pydantic Model.
    def generateAsModel(self, input: str, tolerance = 5, identifier: str = None, last_attempt = False) -> PydanticModel:
        counter, results, output_object = 0, None, None
   
        while counter < tolerance and not output_object and input is not None:
            counter += 1
            try:
                results = self.generate(input = input)
            except OutOfMemoryError as e:
                logger.info(f"Ran out of memory with input of size: {len(input)} on iteration {counter + 1} of {tolerance}. Input:\n{input}")
                output_object = None
                break
            
            output_object = self.outputParser.parse(results)
            

        if (self.resultsfile and identifier and (last_attempt or output_object)):
            json_result = {identifier: output_object.model_dump() if output_object else results if self.debug else None}
            with open(self.resultsfile, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        
        return output_object
    
    