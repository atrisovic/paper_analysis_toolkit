from src.classifier.MistralEnhancedMulticiteClassifier import MistralEnhancedMulticiteClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.prompts.citation_prompts import PROMPT2
from tqdm import tqdm

import warnings, logging
from config import *
import pandas as pd
from torch import cuda, backends, bfloat16

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

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
      
    classifier = MistralEnhancedMulticiteClassifier(model_checkpoint=CITATION_MODEL_PATH,
                                                    llm_model=model,llm_tokenizer=tokenizer, 
                                                    device=device, 
                                                    prompt = PROMPT2)
        
    results_path = '/home/gridsan/afogelson/osfm/paper_analysis_toolkit/citation_intent_mcll_mc_urop_gpt_200'
    df = pd.read_csv(results_path + '.csv')
    
    classifications = [classifier.classify_text(sentence) for sentence in tqdm(df['sentence'])]
    df['mcllmr'] = classifications
    df.to_csv(results_path + '_augmented.csv')   

        
if __name__ == '__main__':
    main()