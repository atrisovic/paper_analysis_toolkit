from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch import cuda, backends
from typing import Tuple

class CitationClassifier:
    def __init__(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        
        device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'


        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)

    def classify_text(self, text) -> Tuple[str, float]:
        results = self.classifier(text, truncation = True)
        as_dictionary = {result['label']: result['score'] for result in results}

        return max(as_dictionary, key=as_dictionary.get)