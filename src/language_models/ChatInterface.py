from pydantic import BaseModel as PydanticModel
from typing import Dict, List
import json
from src.language_models.OutputParser import OutputParser
import logging
from torch.cuda import OutOfMemoryError
import os

username = os.getenv('USER') or os.getenv('USERNAME')
logger = logging.getLogger(__name__)


class ChatInterface:
    def generate(self, input: str, max_new_tokens = 5000, temperature = 1) -> str:
        pass
    
    # Iteratively generate output and force output to match a Pydantic Model.
    def generateAsModel(self, 
                        input: str, 
                        tolerance = 5, 
                        identifier: str = None, 
                        last_attempt = False, 
                        strip_output = True,
                        results_path = None) -> PydanticModel:
        counter, results, output_object = 0, None, None
        
        while counter < tolerance and not output_object and input is not None:
            counter += 1
            try:
                results = self.generate(input = input)
            except OutOfMemoryError as e:
                logger.info(f"Ran out of memory with input of size: {len(input)} on iteration {counter + 1} of {tolerance}. Input:\n{input}")
                output_object = None
                break
            
            output_object = self.outputParser.parse(results, strip_output)
            

        if (results_path and identifier and (last_attempt or output_object)):
            json_result = {identifier: output_object.model_dump() if output_object else results if self.debug else None}
            with open(results_path, 'a+') as f:
                f.write(json.dumps(json_result) + '\n')
        
        return output_object
   

class LlamaCPPChatInterface(ChatInterface):
    def __init__(self, model,
                 outputClass: PydanticModel = None, 
                 debug: bool = True):
        self.model = model
        self.outputClass = outputClass
        self.outputParser = OutputParser(outputClass = outputClass, 
                                         logfile = None)
        self.debug = debug
        
    def get_available_tokens(self, input: str, model, max_context_window: int = 4096) -> int:
        input_bytes = input.encode('utf-8')
        input_length = len(model.tokenize(input_bytes))
        return max(max_context_window - input_length, 0)

    def generate(self, input: str, max_new_tokens = 4096, temperature = 1) -> str:
        # Adjust input length if necessary
        max_new_tokens = 3905 # accounts for other context inputs
        available_tokens = self.get_available_tokens(input, self.model, max_new_tokens)
    
        # Shorten the input if needed
        while available_tokens == 0:
            input = input[:-1]
            available_tokens = self.get_available_tokens(input, self.model, max_new_tokens)

        results = self.model.create_chat_completion(
            messages = [
                {"role": "system", "content": "You are a detailed, knowledgeble, helpful assistant."},
                {
                    "role": "user",
                    "content": input
                }
            ],
            temperature=temperature,
        )
        return results['choices'][0]['message']['content']
    
        
class HFChatInterface(ChatInterface):    
    def __init__(self, model,  tokenizer,  
                 device = None, 
                 outputClass: PydanticModel = None, 
                 debug: bool = True):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.outputClass = outputClass
        self.outputParser = OutputParser(outputClass = outputClass)
        self.debug = debug

    
    # Generate using few shot prompt
    def generate(self, input: str, max_new_tokens = 5000, temperature = 1):        
        encodeds = self.tokenizer.apply_chat_template([{"role": "user", "content": input}], return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, 
                                            max_new_tokens=max_new_tokens, 
                                            pad_token_id = self.tokenizer.eos_token_id, 
                                            do_sample=False, 
                                            temperature=temperature)
        decoded = self.tokenizer.batch_decode(generated_ids)[0]
        
        return decoded
    
    