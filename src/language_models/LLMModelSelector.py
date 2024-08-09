from pydantic import BaseModel as PydanticModel, conlist, model_validator
from src.language_models.FewShot import FewShotPipeline
import logging
from src.prompts.affiliation_prompts import PROMPT1
from typing import List, Literal
import pandas as pd

logger = logging.getLogger(__name__)




class LLMModelSelector():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 device, 
                 models_path: str,
                 debug =  False,
                 strict = True):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.debug = debug
        self.strict = strict
        
        models_df = (pd.read_csv(models_path)
                        .rename(columns = {'Parameters': 'parameters',
                                            'Organization categorization': 'institute_categorizations',
                                            'Notability criteria': 'notability',
                                            'paperId': 'modelId',
                                            'index':'modelKey'}))
    
        self.id_to_keys = {id: set(models_df[models_df['modelId'] == id]['modelKey']) for id in models_df['modelId']}
        
    def find_model_key(self, sentence, modelId):
        modelKeys = self.id_to_keys[modelId]
        if (len(modelKeys) == 1):
            return next(iter(modelKeys))
        else:
            prompt = f"""Each of the following keys represents a different versions of a foundation model: {','.join(modelKeys)}
            The following sentence cites the paper which created the foundation model. Use context clues in the sentence to determine which model was used and return the model key.
            If it is unclear from the context, return NULL. Your response should be exactly one word. """
            return FewShotPipeline(
                    self.model, 
                    self.tokenizer, 
                    prompt = prompt, 
                    device = self.device,
                    debug = self.debug
            ).generate(sentence)

    