from typing import Literal
from src.language_models.FewShot import FewShotPipeline
from pydantic import BaseModel as PydanticModel
from src.language_models.OutputParser import OutputParser
from src.prompts.citation_prompts import *

    
class Classification(PydanticModel):
    classification: Literal['uses', 'extends', 'background', 'motivation', 'future_work', 'differences']
    
class LLMCitationPipeline(FewShotPipeline):
    def __init__(self, model, tokenizer, device = None, prompt: str = PROMPT1):
        super().__init__(model = model, tokenizer=tokenizer, device=device, prompt = prompt, outputClass=Classification)