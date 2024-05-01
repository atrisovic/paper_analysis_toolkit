from pydantic import BaseModel as PydanticModel
from transformers import Pipeline
from typing import Dict, List, Union
from transformers import Pipeline



class FewShotExample(PydanticModel):
    input: str
    output: str
    
    
# just want to log here the reason we didn't continue using the fewshot functionality from langchain
# firstly, it had some strange behvaior with curly brackets that was a bit clunky to get around
# secondly, pipelines don't seem to include instruction tuning, which is pretty darn important
class FewShotPipeline:
    pipeline: Pipeline
    examples: List[Dict]
    
    
    def __init__(self, model, tokenizer, device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.examples = []
        self.device = device
        
    def addExample(self, question: str, answer: Union[str, PydanticModel]):
        if (isinstance(answer, PydanticModel)):
            example = FewShotExample(input = question, output = answer.model_dump_json(indent=1))
        else:
            example = FewShotExample(input = question, output = answer)
            
        self.examples.append(example)   
        
    def getSchema(self):
        pass
    
    def wrapInstructions(self, input: str):
        pass   
    
    def generateZeroShotPrompt(self, input: str, output: str = None):
        zero_shot = [{"role": "user", "content": self.wrapInstructions(input)}]
        
        if output:
            zero_shot.append({"role": "assistant", "content": output})
        
        return zero_shot

    def getFewShotPrompt(self, input: str, max_examples: int = None):
        few_shots = []
        for example in self.examples[:max_examples]:
            few_shots += self.generateZeroShotPrompt(example.input, example.output)
            
        few_shots += self.generateZeroShotPrompt(input = input, output = None)
            
        return few_shots
        
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
    
    
    