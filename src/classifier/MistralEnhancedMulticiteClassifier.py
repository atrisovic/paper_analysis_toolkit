from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch import cuda, backends
from typing import Tuple
from src.classifier.CitationClassifier import CitationClassifier
from src.language_models.LLMCitations import LLMCitationPipeline, Classification


class MistralEnhancedMulticiteClassifier(CitationClassifier):
    def __init__(self, model_checkpoint, llm_model, llm_tokenizer, device = None):    
        device = device or ('mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu')
        self.mistral_pipeline = LLMCitationPipeline(model = llm_model, tokenizer=llm_tokenizer, device = device)

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.multicite_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)
        
    def classify_text(self, text, *args) -> Tuple[str, float]:
        results = self.multicite_classifier(text, truncation = True)
        as_dictionary = {result['label']: result['score'] for result in results}

        best_label = max(as_dictionary, key=as_dictionary.get)
        
        if (best_label in ('uses', 'extends')):
            llm_label: Classification = self.mistral_pipeline.generateAsModel(input = text)
            if (llm_label is not None):
                return llm_label.classification
            else:
                pass
        
        return best_label
    
    def getClassificationOrdering(self):
        return ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']
    