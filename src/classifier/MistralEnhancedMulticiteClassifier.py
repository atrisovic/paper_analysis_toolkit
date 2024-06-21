from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch import cuda, backends
from typing import Tuple
from src.classifier.CitationClassifier import CitationClassifier
from src.language_models.LLMCitations import LLMCitationPipeline, Classification
from src.language_models.LLMBinaryCitations import LLMBinaryCitationPipeline, BinaryClassification
from src.prompts.citation_prompts import PROMPT1, PROMPT3
from torch.cuda import OutOfMemoryError
import logging

logger = logging.getLogger(__name__)

class MistralEnhancedMulticiteClassifier(CitationClassifier):
    def __init__(self, model_checkpoint, llm_model, llm_tokenizer, device = None, prompt: str = PROMPT1, mc_uses_extends = False):    
        device = device or ('mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu')
        self.mistral_pipeline = LLMCitationPipeline(model = llm_model, tokenizer=llm_tokenizer, device = device, prompt = prompt)

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.multicite_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)
        self.mc_uses_extends = mc_uses_extends
        
    def classify_text(self, text, *args) -> Tuple[str, float]:
        results = self.multicite_classifier(text, truncation = True)

        as_dictionary = {result['label']: result['score'] for result in results}

        best_label = max(as_dictionary, key=as_dictionary.get)
        
        if (best_label in ('uses', 'extends')):
            llm_label: Classification = self.mistral_pipeline.generateAsModel(input = text)
            if (llm_label is not None and ((llm_label.classification not in {'uses', 'extends'}) ** self.mc_uses_extends)):
                return llm_label.classification
        
        return best_label
    
    def getClassificationOrdering(self):
        return ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']

    

class MistralEnhancedMulticiteClassifierBinary(CitationClassifier):
    def __init__(self, model_checkpoint, llm_model, llm_tokenizer, device = None, prompt: str = PROMPT3):    
        device = device or ('mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu')
        self.mistral_pipeline = LLMBinaryCitationPipeline(model = llm_model, tokenizer=llm_tokenizer, device = device, prompt = prompt)

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.multicite_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)
        
    def classify_text(self, text, *args) -> Tuple[str, float]:
        llm_label: BinaryClassification = self.mistral_pipeline.generateAsModel(input = text)
        return None if llm_label is None else (llm_label.classification, llm_label.explanation)
    
    
        results = self.multicite_classifier(text, truncation = True)
        as_dictionary = {result['label']: result['score'] for result in results}

        best_label = max(as_dictionary, key=as_dictionary.get)
        
        if (best_label in ('uses', 'extends')):
            llm_label: BinaryClassification = self.mistral_pipeline.generateAsModel(input = text)
            if (llm_label is not None):
                return (llm_label.classification, llm_label.explanation)
        
        return best_label
    
    def getClassificationOrdering(self):
        return ['uses', 'context']
    
    
class MistralEnhancedMulticiteClassifierFreeform(CitationClassifier):
    def __init__(self, model_checkpoint, llm_model, llm_tokenizer, device = None, prompt: str = PROMPT1):    
        device = device or ('mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu')
        self.mistral_pipeline = LLMCitationPipeline(
                                    model = llm_model, 
                                    tokenizer=llm_tokenizer, 
                                    device = device, 
                                    prompt = prompt, 
                                    outputClass = dict
                                )

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.multicite_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)
        
    def classify_text(self, text, *args) -> Tuple[str, float]:
        return self.mistral_pipeline.generateAsModel(input = text)

    def getClassificationOrdering(self):
        return ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']