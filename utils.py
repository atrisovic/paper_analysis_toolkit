
from os import walk, path
import regex as re
from nltk.tokenize import sent_tokenize
from typing import List
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from tqdm import tqdm
import pickle
import json
import pandas as pd
from math import ceil
from torch.backends import mps
from torch import cuda
#nltk.download('punkt')
from typing import List, Dict, Tuple

def soft_assert(condition, statement):
    if not condition:
        print(f"SOFT ASSERT FAILURE: {statement}")


class CitationClassifier:
    def __init__(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        
        device = 'mps' if mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'
        
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = device)

    def classify_text(self, text) -> Tuple[str, float]:
        results = self.classifier(text, truncation = True)[-1]
        return results['label'], results['score']


class TextualReference:
    def __init__(self, sentence):
        self.sentence: str = sentence
        self.classification: str = None
        self.score: float = None
        
    def classify(self, classifier: CitationClassifier):
        self.classification, self.score = classifier.classify_text(self.sentence)
    
    def as_dictionary(self):
        order_class_values = ['extends', 'uses', 'differences', 'similarities', 'future_work', 'motivation', 'background']
        classification_rankings = {val: idx for idx, val in enumerate(order_class_values)}
        return {
                'sentence': self.sentence,
                'classification': self.classification,
                'score': self.score,
                'classification_order': classification_rankings[self.classification]
                }


class Reference:
    def __init__(self, title, key, citation = None):
        self.title: str = title.lower()
        self.key: str = key.lower()
        self.citation: str = citation
        self.textualReferences: List[TextualReference] = None
        
    def getCitationFromContent(self, content: str, ref_data = None) -> str:
        numerical_refs = re.findall(f"\* (\[\d+\]).*{self.title}", content) #  * [38] SOME TEXT HERE title
        string_refs = re.findall(f"\* ([^\\n\)]+[\)\]]).*{self.title}", content) #  * Name, Extra, (Year) SOME TEXT HERE title

        error_string = f"Title: {self.title}; Content: {content[:50]}...; numerical_ref: {numerical_refs}; string_ref: {string_refs}, ref_data: {ref_data}"
        
        soft_assert(self.title not in content or numerical_refs or string_refs, error_string)

        ref_number = None if not numerical_refs else numerical_refs[0]
        ref_str = None if not string_refs else re.sub(r'\s*[\(\[]?(\d+)[\)\]]?', r', \1', string_refs[0])

        self.citation = ref_number or ref_str
        
        return self.citation
    
    def getSentencesFromContent(self, all_sentences: List[str]) -> List[str]:
        if self.citation is None:
            self.textualReferences = []
        else:
            self.textualReferences = [TextualReference(sentence) for sentence in all_sentences if (self.citation in sentence) and (sentence[0] != '*')]
            
        return self.textualReferences

    def classifyAllSentences(self, classifier: CitationClassifier):
        for textualReference in self.textualReferences:
            textualReference.classify(classifier=classifier)
            
    def textualReferencesToDictList(self):
        return [textRef.as_dictionary() | {'FM_key': self.key, 'FM_title': self.title} for textRef in self.textualReferences]


class Paper:
    def __init__(self, path: str):
        with open(path, "r") as file:
            file_content = file.read()

        self.path: str = path
        
        first_line = file_content.split('\n')[0]
        self.title: str = re.sub(r'#','', first_line)
        
        self.content: str = file_content.lower()
        
        self.all_sentences: List[str] = sent_tokenize(self.content)
        
        self.references: Dict[str, Reference] = {}
        
    def getReferenceFromTitle(self, title, key, classifier = None) -> Reference:
        reference = Reference(title = title, key = key)
        reference.getCitationFromContent(content = self.content, ref_data = self.path)
        reference.getSentencesFromContent(all_sentences=self.all_sentences)
        
        if classifier:
            reference.classifyAllSentences(classifier = classifier)

        self.references[key] = reference
        
        return reference
    
    def textualReferencesToDictList(self):
        return [row | {'paper': self.title} for title, reference in self.references.items() for row in reference.textualReferencesToDictList()]
            

class Corpus:
    def __init__(self, directory, extensions, limit = 1000, cluster = None, cluster_count = None):
        self.limit = limit
        self.directory = directory
        self.extensions = extensions
        self.cluster = cluster
        self.cluster_count = cluster_count
        self.papers: List[Paper] = self.discoverPapers()

    def discoverPapers(self) -> List[Paper]:
        papers = []
        for root, dirs, files in walk(self.directory):
            papers += [Paper(path.join(root, file))
                                    for file in files
                                        if (file.split('.')[-1] in self.extensions)]
        if (self.limit or self.cluster):
            papers = sorted(papers, key = lambda s: s.title) # for consistency when clustering
            
        if (self.limit):
            papers = papers[:self.limit]
            
        if self.cluster:
            chunk_length = ceil(len(papers)/self.cluster_count)
            papers = papers[chunk_length * self.cluster: chunk_length * (self.cluster + 1)]
            
        self.papers = papers
        return self.papers
        

    def findAllPaperReferencesByTitle(self, title: str, key: str, classifier: CitationClassifier):
        for paper in self.papers:
            paper.getReferenceFromTitle(title, key, classifier = classifier)
        
    def textualReferencesToDictList(self):
        return [row for paper in self.papers for row in paper.textualReferencesToDictList()]


if __name__ == '__main__':
    classifier = CitationClassifier('allenai/multicite-multilabel-scibert')
    corpus = Corpus('./Markdown', extensions = ['mmd'], limit = 1000)
    
    with open('121_results_v2.json', 'r') as f:
        foundational_models_json = json.load(f)

    for key, data in tqdm(foundational_models_json.items()):
        title = re.escape(data['title']) # one of the paper titles has a backslash in it. not a permanent solution!
        corpus.findAllPaperReferencesByTitle(title = title, key = key, classifier = classifier)

    df = pd.DataFrame.from_dict(corpus.textualReferencesToDictList())
    
    df['classification_ranking'] = df.groupby(['FM_key', 'paper'])['classification_order'].rank(method='min')
    foundationalModelUseCaseCounts = (df[df['classification_ranking'] == 1]
                                            .groupby(['FM_key', 'classification'])['paper']
                                            .nunique()
                                            .reset_index()
                                            .rename(columns={'paper':'count'})
                                        )    
    

    with open('pickle/df.pkl', 'wb') as f:
        pickle.dump(df, f)
        
    with open('pickle/corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
        
    with open('pickle/foundationalModelUseCaseCounts.pkl', 'wb') as f:
        pickle.dump(foundationalModelUseCaseCounts, f)