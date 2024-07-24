from pydantic import BaseModel as PydanticModel
from typing import Dict, List
import json
from src.language_models.OutputParser import OutputParser
import logging
from torch.cuda import OutOfMemoryError

logger = logging.getLogger(__name__)


class ChatInterface:    
    def __init__(self, model,  tokenizer,  
                 device = None, 
                 outputClass: PydanticModel = None, 
                 resultsfile: str = None,
                 debug: bool = True):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.outputClass = outputClass
        self.outputParser = OutputParser(outputClass = outputClass, 
                                         logfile = '/home/gridsan/afogelson/osfm/paper_analysis_toolkit/temp.log')
        self.resultsfile = resultsfile
        self.debug = debug

    
    # Generate using few shot prompt
    def generate(self, input: str, max_new_tokens = 5000, temperature = 1):                
        encodeds = self.tokenizer.apply_chat_template(input, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(model_inputs, 
                                            max_new_tokens=max_new_tokens, 
                                            pad_token_id = self.tokenizer.eos_token_id, 
                                            do_sample=True, 
                                            temperature=temperature)
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
    