from src.process.Corpus import Corpus
from src.process.Cluster import Cluster
from datetime import datetime 
from config import MARKDOWN_FILES_PATH, LLM_MODEL_NAME, LLM_MODEL_PATH, LLM_TOKENIZER_PATH
import  logging, argparse

from paper_analysis_toolkit.src.language_models.InstitutionsPipeline import InstitutionsPipeline, ListInstitutions
from src.prompts.affiliation_prompts import PROMPT3
from src.language_models.ChatInterface import LlamaCPPChatInterface

from llama_cpp import Llama


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default = 1, type = int, help = 'One-indexed index of worker for this particular job.')
    parser.add_argument('-n', '--workers', default = 1, type = int, help = 'Total jobs to be run *in separate jobs*')
    parser.add_argument('-l', '--limit', type = int, help = 'Limit the number of documents scanned.')
    parser.add_argument('-f', '--filter_file', type = str, help = 'A list of files to be included in the corpus (others from directory will be discarded).')
    parser.add_argument('-d', '--debug', action = 'store_true', help = "Adding this flag will enabled debug logging.")
    parser.add_argument('-s', '--seed', default = 0, type = int, help = "Seed used for all random processes. Default is 0.")
    parser.add_argument('--eagerstorage', action = 'store_true', help = "Adding this flag will decrease RAM usage but increase runtime when rereading documents.")

    args = parser.parse_args()
    
    cluster = Cluster(index = args.index, worker_count = args.workers, limit = args.limit, seed = args.seed)
    
    right_now = datetime.now().replace(microsecond=0)
    logfile = f"logs/affiliations/logfile_{right_now}_worker{args.index}of{args.workers}.log"
    results_path = f"results/affiliations/results_{right_now}_worker{args.index}of{args.workers}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG if args.debug else logging.INFO)
    

    model_path = 'saved_models/models--bullerwins--Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf'
    model = Llama(model_path, n_gpu_layers = -1, n_ctx = 4096, verbose = False)
    interface = LlamaCPPChatInterface(model = model, outputClass = ListInstitutions)
    
    affPipepline = InstitutionsPipeline(interface = interface,
                                    prompt = PROMPT3,
                                    debug = args.debug)
        
    corpus = Corpus(MARKDOWN_FILES_PATH, 
                        extensions = ['mmd'], 
                        cluster = cluster,
                        filter_path = args.filter_file,
                        lazy = not args.eagerstorage,
                        confirm_paper_ref_sections=False)
    
    corpus.getAllAffiliations(pipeline = affPipepline, results_path=results_path)



if __name__ == '__main__':
    main()
