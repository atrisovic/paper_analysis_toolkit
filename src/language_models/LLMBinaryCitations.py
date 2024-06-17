from typing import Literal
from src.language_models.FewShot import FewShotPipeline
from pydantic import BaseModel as PydanticModel
from src.language_models.OutputParser import OutputParser
from src.prompts.citation_prompts import *

    
class BinaryClassification(PydanticModel):
    explanation: str
    classification: Literal['uses', 'context', 'unclear']
    
class LLMBinaryCitationPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, prompt: str, device = None):
        super().__init__(model = model, tokenizer=tokenizer, device=device, prompt = prompt, outputClass=BinaryClassification)