from typing import Literal
from src.language_models.FewShot import FewShotPipeline
from pydantic import BaseModel as PydanticModel
from src.language_models.OutputParser import OutputParser
from src.prompts.citation_prompts import *
from src.language_models.ChatInterface import HFChatInterface

    
class Classification(PydanticModel):
    classification: Literal['uses', 'extends', 'background', 'motivation', 'future_work', 'differences']
    
class LLMCitationPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, prompt: str, device = None, outputClass = Classification, debug: bool = False):
        interface = HFChatInterface(model, tokenizer, device, Classification, debug = debug)
        super().__init__(interface = interface, prompt = prompt, outputClass=outputClass)