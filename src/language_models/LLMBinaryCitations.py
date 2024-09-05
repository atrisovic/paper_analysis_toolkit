from typing import Literal
from src.language_models.FewShot import FewShotPipeline
from pydantic import BaseModel as PydanticModel
from src.language_models.OutputParser import OutputParser
from src.prompts.citation_prompts import *
from src.language_models.ChatInterface import HFChatInterface
    
class BinaryClassification(PydanticModel):
    explanation: str
    classification: Literal['uses', 'context', 'unclear']
    
class LLMBinaryCitationPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, prompt: str, device = None, debug: bool = False, resultsfile: str = None):
        interface = HFChatInterface(model, tokenizer, device, BinaryClassification, debug = debug)
        super().__init__(interface = interface, prompt = prompt, outputClass=BinaryClassification, resultsfile = resultsfile, debug = debug)