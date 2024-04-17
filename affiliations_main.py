from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import backends, cuda, bfloat16
from affiliations.AffiliationClassifier import AffiliationClassifier
from documents.Corpus import Corpus
from datetime import datetime 
from config import MARKDOWN_FILES_PATH, LLM_MODEL_NAME, LLM_MODEL_PATH, LLM_TOKENIZER_PATH
import nltk, logging, argparse
nltk.download('punkt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of documents scanned.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('--eagerstorage', action = 'store_true', help = "Adding this flag will decrease RAM usage but increase runtime when rereading documents.")

    args = parser.parse_args()
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/affiliations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    resultsfile = f"results/affiliations/results_{right_now}_worker{args.index}of{args.workers}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)
    
    device = 'mps' if backends.mps.is_available() else 'cuda' if cuda.is_available() else 'cpu'

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=bfloat16)


    refresh = False
    try:
        assert(not refresh)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, device_map = device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_PATH)
    except:
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map=device, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        model.save_pretrained(LLM_MODEL_PATH, from_pt=True)
        tokenizer.save_pretrained(LLM_TOKENIZER_PATH, from_pt = True)
        

    aff_classifier = AffiliationClassifier(model, tokenizer, device)
    corpus = Corpus(MARKDOWN_FILES_PATH, 
                        extensions = ['mmd'], 
                        cluster_info = (args.index, args.workers), 
                        paper_limit = args.limit, 
                        filter_path = args.filter_file,
                        lazy = not args.eagerstorage,
                        confirm_paper_ref_sections=False)
    corpus.setAllAffiliations(classifier = aff_classifier, resultsfile = resultsfile)



if __name__ == '__main__':
    main()