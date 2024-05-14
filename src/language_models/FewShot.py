from pydantic import BaseModel as PydanticModel
from transformers import Pipeline
from typing import Dict, List, Union
from transformers import Pipeline
import json
from src.language_models.OutputParser import OutputParser

class FewShotExample(PydanticModel):
    input: str
    output: str
    
    
    
class FewShotPipeline:
    pipeline: Pipeline
    examples: List[Dict]
    
    def __init__(self, model, tokenizer, prompt: str, device = None, outputClass: PydanticModel = None, resultsfile: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.examples = []
        self.device = device
        self.outputParser = OutputParser(outputClass = outputClass)
        self.resultsfile = resultsfile
        self.prompt = prompt
        
        for question, answer in self.getExamples():
            self.addExample(question=question, answer=answer)
    
    # Implemented in children
    def getExamples(self):
        return []
    
    def wrapInstructions(self, input: str):
        self.prompt.format(input = input)   

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
    def generateAsModel(self, input: str, tolerance= 5, identifier: str = None) -> PydanticModel:
        counter = 0
        output_object = None
        
        while counter < tolerance and not output_object:
            counter += 1
            results = self.generate(input = input)
            output_object = self.outputParser.parse(results)
            

        if (self.resultsfile and output_object and identifier):
            json_result = {identifier: output_object.model_dump()}
            with open(self.resultsfile, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        elif (self.resultsfile and identifier):
            json_result = {identifier: None}
            with open(self.resultsfile, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        
        return output_object
    
    