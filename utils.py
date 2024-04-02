
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
#nltk.download('punkt')


class Document:
    def __init__(self, path: str):
        with open(path, "r") as file:
            file_content = file.read()

        self.path = path
        self.title = path # temporary, will fix in a bit
        self.text = file_content.lower()
        self.sentences = sent_tokenize(self.text)


    ''' Filters all sentences by a particular ref '''
    def citations_from_ref(self, ref: str) -> List[str]:
        return [] if ref is None else [sentence for sentence in self.sentences if (ref in sentence) and (sentence[0] != '*')]


    ''' Retrieves appropriate citation based on paper title'''
    def ref_from_title(self, title: str) -> str:
        title = title.lower()
        
        numerical_refs = re.findall(f"\* (\[\d+\]).*{title}", self.text)
        string_refs = re.findall(f"\* ([^\\n\)]+[\)]).*{title}", self.text)

        if not (len(numerical_refs) <= 1 and len(string_refs) <= 1):
            print(f"{self.path} seems to contain the following title twice: {title}")

        ref_number = None if not numerical_refs else numerical_refs[0]
        ref_str = None if not string_refs else re.sub(r'\s*[\(\[]?(\d+)[\)\]]?', r', \1', string_refs[0])

        return ref_number or ref_str


    ''' Retrieves all citation sentences in a document based on the title. '''
    def citations_from_title(self, title: str) -> List[str]:
        ref = self.ref_from_title(title = title)
        return self.citations_from_ref(ref)


class CitationClassifier:
    def __init__(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length = 512)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

    def classify_sentence(self, sentence) -> List[dict]:
        label = self.classifier(sentence, truncation = True)[-1]
        label['sentence'] = sentence
        return label

    def classify_document_citations(self, document: Document, title: str) -> List[dict]:
        sentences = document.citations_from_title(title)
        labels = []
        for label in map(self.classify_sentence, sentences):
            label['foundation_model'] = title
            label['paper'] = path.basename(document.title) # will update this not to use the basename, but the title, when i get there!
            labels.append(label)
        return labels
    
class Corpus:
    def __init__(self, directory, extensions, limit = 10):
        self.limit = limit
        self.directory = directory
        self.extensions = extensions
        self.documents = self.find_documents()

    def find_documents(self):
        documents = []
        for root, dirs, files in walk(self.directory):
            documents += [Document(path.join(root, file))
                                    for file in files
                                        if (file.split('.')[-1] in self.extensions)]
        return documents[:self.limit]

    def classify_citations_from_title(self, classifier: CitationClassifier, title: str) -> List[dict]:
        results = []
        for document in self.documents:
            results += classifier.classify_document_citations(document, title)
        return results




markdown_path = './Markdown'
corpus = Corpus(markdown_path, extensions = ['mmd'])
classifier = CitationClassifier('allenai/multicite-multilabel-scibert')

with open('121_results_v2.json', 'r') as f:
    foundation_models_json = json.load(f)

titles = [re.escape(paper['title']) for _, paper in foundation_models_json.items()] # one of the paper titles has a backslash in it. not a permanent solution!

results: List[dict] = []
for title in tqdm(titles):
    results += corpus.classify_citations_from_title(classifier, title)

df = pd.DataFrame.from_dict(results).groupby(['foundation_model', 'label']).size()


# this is just for demo purposes, not used in the script
with open('results.pkl', 'wb') as f:
    pickle.dump(df, f)
    
with open('results.pkl', 'rb') as f:
    results = pickle.load(f) 