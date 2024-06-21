from src.classifier.MistralEnhancedMulticiteClassifier import MistralEnhancedMulticiteClassifierFreeform
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.prompts.citation_prompts import PROMPT8
from tqdm import tqdm

import warnings, logging
from config import *
import pandas as pd
from torch import cuda, backends, bfloat16
from datetime import datetime

logger = logging.getLogger(__name__)

logging.basicConfig(filename=f"/home/gridsan/afogelson/osfm/paper_analysis_toolkit/logs/scripts/bin_sample_{datetime.now()}.csv",
                    level=logging.DEBUG)

def main():            
    device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'
    print(f"Using device = {device}")
    
    
    bnb_config = None if device != 'cuda:0' else BitsAndBytesConfig(load_in_4bit=True,
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
      
    classifier = MistralEnhancedMulticiteClassifierFreeform(model_checkpoint=CITATION_MODEL_PATH,
                                                    llm_model=model,
                                                    llm_tokenizer=tokenizer, 
                                                    device=device, 
                                                    prompt = PROMPT8)
        
    results_path =  '/home/gridsan/afogelson/osfm/scripts/urop_samples/uniform_sample/uniform_urop_sample_alex_labeled'
    df = pd.read_csv(results_path + '.csv')
    
    classifications = []
    for idx, sentence in enumerate(tqdm(df['multisentence'])):
        if (True):
            classifications.append(classifier.classify_text(sentence))
        else:
            classifications.append(None)

    df['mcllm_binary'] = classifications
    df.to_csv(results_path + '_PROMPT8.csv')   

        
if __name__ == '__main__':
    main()