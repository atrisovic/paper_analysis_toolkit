from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import backends, cuda, bfloat16
from affiliations.AffiliationClassifier import AffiliationClassifier
from documents.Corpus import Corpus
import nltk
nltk.download('punkt')
import argparse
from datetime import datetime
import logging

def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Max amount of documents to process. Default is None.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    
    args = parser.parse_args()
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/affiliations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    resultsfile = f"results/affiliations/results_{right_now}_worker{args.index}of{args.workers}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)
    
    device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'

    markdown_file_path = './data/Markdown/'

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    local_model_path = './saved_models/mistral_model'
    local_tokenizer_path = './saved_models/mistral_tokenizer'


    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=bfloat16)

    refresh = False
    try:
        assert(not refresh)
        model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map = device)
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.save_pretrained(local_model_path, from_pt=True)
        tokenizer.save_pretrained(local_tokenizer_path, from_pt = True)
        

    aff_classifier = AffiliationClassifier(model, tokenizer, device)
    corpus = Corpus(markdown_file_path, 
                        extensions = ['mmd'], 
                        cluster_info = (args.index, args.workers), 
                        limit = args.limit, 
                        filter_path = args.filter_file)
    corpus.setAllAffiliations(classifier = aff_classifier, resultsfile = resultsfile)



if __name__ == '__main__':
    main()