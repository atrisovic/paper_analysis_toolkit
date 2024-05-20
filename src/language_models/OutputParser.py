from pydantic import BaseModel as PydanticModel
import json
from typing import Optional


class OutputParser:
    def __init__(self, outputClass: PydanticModel):
        self.outputClass = outputClass
    
    def stripOutput(self, text: str, begin_token: str = '[/INST]', end_token = '</s>'):
        begin_index = text.rfind(begin_token)
        text = text[begin_index + len(begin_token):] if begin_index >= 0 else text
        
        
        end_index = text.rfind(end_token)
        text = text[:end_index] if end_index >= 0 else text
        
        return text
    
    
    def stripJSON(self, text: Optional[str], added_bracket = False) -> Optional[dict]:
        if (text is None):
            return None
        
        text = text.replace('\(', '(').replace('\)', ')') #Mistral likes to add escape sequences, for some unknown reason
        # a very manual way of finding our JSON string within the output
        all_open_brackets = [i for i, ltr in enumerate(text) if ltr == '{']
        obj = None
        for start in all_open_brackets:
            balance_counter = 1
            for offset, chr in enumerate(text[start + 1:], start = 1):
                balance_counter += (1 if chr == '{' else -1 if chr == '}' else 0)
                if (balance_counter == 0):
                    try:
                        obj = json.loads(text[start: start + offset + 1])
                    except:
                        pass
                    break
            if (obj is not None):
                break
            
        # sometimes we miss the first or last bracket (dumb LLM), so we add it manually.
        if not added_bracket:
            return (self.stripJSON('{' + text, added_bracket = True) or self.stripJSON(text + '}', added_bracket = True))
            
        return obj
    
    def stripModel(self, obj: Optional[dict]) -> Optional[PydanticModel]:
        if (obj is None):
            return None
        
        try:
            return self.outputClass(**obj)
        except:
            return None
        
    def parse(self, text: str) -> Optional[PydanticModel]:
        output = self.stripOutput(text)
        json_obj = self.stripJSON(output)
        model = self.stripModel(json_obj)
        
        return model