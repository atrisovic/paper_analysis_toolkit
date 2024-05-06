import pandas as pd
from classes.CitationClassifier import MistralEnhancedMulticiteClassifier, MultiCiteClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import *
import pandas as pd
from torch import cuda, backends, bfloat16
from tqdm import tqdm

df = pd.read_csv('data/task_2_model_reuse.csv').rename(columns = {'How does it use <model.name> ?': 'manual'})

df = df[df["Citation Sentence"].notna()]
df = df.dropna(subset=['manual'])
df.head(3)


device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'
print(f"Using device = {device}")
bnb_config = None if device != 'cuda' else BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_compute_dtype=bfloat16) 

refresh = False
try:
    assert(not refresh)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device_map = device, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_PATH, device = device)
except:
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map=device, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, device = device)

    model.save_pretrained(LLM_MODEL_PATH, from_pt=True)
    tokenizer.save_pretrained(LLM_TOKENIZER_PATH, from_pt = True)
    
enhanced_multicite = MistralEnhancedMulticiteClassifier(model_checkpoint=CITATION_MODEL_PATH,llm_model=model,llm_tokenizer=tokenizer, device=device)
multicite_classifier = MultiCiteClassifier(model_checkpoint=CITATION_MODEL_PATH)


classifications = []
for sentence in tqdm(df['Citation Sentence']):  
    result = enhanced_multicite.classify_text(sentence)
    classifications.append(result)
    
df['MCLLM'] = classifications




classifications = []
for sentence in tqdm(df['Citation Sentence']):  
    result = multicite_classifier.classify_text(sentence)
    classifications.append(result)
    
df['MC'] = classifications



df.to_csv('mcllm_cm.csv')