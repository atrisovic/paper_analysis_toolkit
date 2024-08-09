from src.classifier.MistralEnhancedMulticiteClassifier import MistralEnhancedMulticiteClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.prompts.citation_prompts import PROMPT2

from config import *
from torch import cuda, backends, bfloat16

def get_device():
    return 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'

def load_saved_model():
    device = get_device()
    print(f"Using device = {device}")

    bnb_config = None #if 'cuda' not in device else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16) 
    
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device_map = device, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_PATH, device = device)

        
    return model, tokenizer

def download_model():
    device = get_device()
    print(f"Using device = {device}")

    bnb_config = None if 'cuda' not in device else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16) 
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map=device, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, device = device)
    model.save_pretrained(LLM_MODEL_PATH, from_pt=True)
    tokenizer.save_pretrained(LLM_TOKENIZER_PATH, from_pt = True)

def load_classifier(prompt):    
    device = get_device()

    model, tokenizer = load_saved_model()

    classifier = MistralEnhancedMulticiteClassifier(model_checkpoint=CITATION_MODEL_PATH,
                                                    llm_model=model,
                                                    llm_tokenizer=tokenizer, 
                                                    device=device, 
                                                    prompt = prompt,
                                                    mc_uses_extends=True)
    
    return classifier
