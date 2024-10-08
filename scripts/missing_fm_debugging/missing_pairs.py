import jsonlines
from pydantic import BaseModel
from typing import List
from json import dumps
from src.document.paper import Paper
from src.process.FoundationModel import FoundationModel
from os.path import join
from tqdm import tqdm
import numpy as np

CITATION_GRAPH_PATH = '/home/gridsan/afogelson/osfm/paper_analysis_toolkit/scripts/missing_fm_debugging/citation_graph_missing_mk_present_0927.jsonl'
DATA_PATH = '/home/gridsan/afogelson/osfm/data'


class MissingFoundationModel(BaseModel):
    paperId: str
    citingPapers: List[str]
    modelKey: str
    title: str
    
class MissingPair(BaseModel):
    modelId: str
    paperId: str
    modelKey: str
    title: str 
    missing_reference: int = None
    missing_sentences: int = None
    
def getAllMissingPairs():
    pairs = []
    with jsonlines.open(CITATION_GRAPH_PATH) as reader:
        for obj in reader:
            fm = MissingFoundationModel.model_validate_json(dumps(obj))
            pairs += [MissingPair(modelId = fm.paperId, paperId = paperId, modelKey = fm.modelKey, title = fm.title) for paperId in fm.citingPapers]
    return pairs


def check_missing(missing_pair, debug = False):
    fm = FoundationModel(key = missing_pair.modelKey, title= missing_pair.title, id= missing_pair.modelId, year=None)
    paper_path = join(DATA_PATH,f'markdown/{missing_pair.paperId}.mmd')
    sus_paper = Paper(paper_path, confirm_reference_section = False)
    ref = sus_paper.getReferenceFromTitle(fm)
    missing_pair.missing_sentences = 1 if ref.reference_exists and len(ref.textualReferences) == 0 else 0
    missing_pair.missing_reference = 1 if ref.reference_might_exist and not ref.reference_exists else 0

    if (debug or missing_pair.missing_reference == 1):
        print(missing_pair, ref)
        pass
    return missing_pair


def checkAllPairs():
    pairs = getAllMissingPairs()
    missing = []
    for idx, missing_pair in tqdm(enumerate(pairs), total = len(pairs)):
        if (missing_pair.modelKey not in current_missing_fms()):
            continue
        missing_pair = check_missing(missing_pair)
        pairs[idx] = missing_pair
        
        missing.append(missing_pair.missing_reference)
        if idx % 100 == 0:
            print(np.array(missing).sum()/len(missing))
            
             
def diagnose_miss(modelId: str, paperId: str, modelKey: str, title: str):
    pair = MissingPair(modelId = modelId, paperId = paperId, modelKey = modelKey, title = title)
    pair = check_missing(pair, debug = True)
    return pair
    
def current_missing_fms():
    return {'473_dna_fine-tuned_language_model_(dflm)', '948_doc_+_finetune∗_+_partial_shuffle_(wt2)', '548_ctm_(cifar-10)', '357_ei-rehn-1000d', '1204_pixart-α', '1201_kwaiyiimath', '1162_baai_bge-reranker-large', '1285_mistral_7b_+_ovm', '1100_multiband_diffusion', '1215_weblab-10b', '527_$\\infty$-former_(sm)', '1159__tinyllama-1.1b_(3t_tokens_checkpoint)', '1149_multi-cell_lstm', '1200_xinghan_foundation_model', '705_mms-1b', '469_qwen-audio-chat', '794_binarized_neural_network_(mnist)', '25_gpt-sw3', '1238_decaying_fast_weights_transformer_(wt-103)', '1198_deepsa', '421_inflated_3d_convnet', '889_pangu-σ', '1185_gemini_1_pro', '612_solar-10.7b', '368_robocat', '380_prott5-xxl', '677_memsizer', '1188_granite_13b', '68_calm', '793_gpt-2_(fine-tuned_with_hydra)', '500_lep-ad', '1109_pangu-α', '951_retrieval-quality-knn-lms', '902_sparse_wide_gpt-3_small', '1251_dall-e_mini', '1199_qmoe:_compressed_1t_model', '1277_cancer_drug_mechanism_prediction', '1126_decaying_fast_weights_transformer', '800_omnivec', '746_llama_guard', '281_seamlessm4t', '1195_prithvi-100m', '663_4-gram_+_8_denn', '153_wenet_(wt2)', '240_bluumi', '1203_tinyllama-1.1b_(3t_token_checkpoint)', '275_b2t_connection_(16l)', '633_rnnlm_+_dynamic_kl_regularization_(wt2)', '1276_deep_blue', '107_hyperclova', '290_alphageometry', '1040_gl-lwgc-awd-mos-lstm_+_dynamic_evaluation_(wt2)', '1193_volcano_13b', '98_gpt2-layerfusion-ws', '319_codefuse-13b', '694_egru_(wt2)', '590_rt-trajectory', '1214_claude_instant', '1209_bge-reranker-large', '973_fingpt-13b', '883_skywork-13b', '1181_poro34b_(700b_token_checkpoint)', '1205_plamo-13b', '1245_engine-xl(ne)', '1023_pointnet++', '820_nüwa', '1210_mobile_v-moes', '554_equidock', '425_deepseekmoe-16b', '571_codet5+', '659_vrns-rnn-3-3-5', '1196_otterhd-8b', '1186_onellm', '1211_falcon-180b', '1133_gpt2+corelm+fine-tuning', '1241_dall-e_mega', '583_table-gpt', '1089_wenet_(penn_treebank)', '1212_refact-1.6b', '1180_videopoet', '1074_lstm-large+behaviorial-gating', '1206_emu_(meta)', '736_fold2seq', '627_flm-101b', '1022_pagnol-xl', '595_dcn+', '813_segatron-xl_large,_m=384_+_hcp', '1011_ankh_large', '927_ankh_base', '1190_starling-lm-7b-alpha'}


{'469_qwen-audio-chat', '500_lep-ad', '98_gpt2-layerfusion-ws', '951_retrieval-quality-knn-lms', '1203_tinyllama-1.1b_(3t_token_checkpoint)', '1109_pangu-α', '68_calm', '627_flm-101b', '705_mms-1b', '290_alphageometry', '1011_ankh_large', '275_b2t_connection_(16l)', '659_vrns-rnn-3-3-5', '583_table-gpt', '527_$\\infty$-former_(sm)', '1196_otterhd-8b', '421_inflated_3d_convnet', '813_segatron-xl_large,_m=384_+_hcp', '820_nüwa', '1023_pointnet++', '1180_videopoet', '571_codet5+', '736_fold2seq', '794_binarized_neural_network_(mnist)', '1211_falcon-180b', '554_equidock', '1209_bge-reranker-large', '25_gpt-sw3', '1195_prithvi-100m', '1277_cancer_drug_mechanism_prediction', '1186_onellm', '1022_pagnol-xl', '800_omnivec', '380_prott5-xxl', '793_gpt-2_(fine-tuned_with_hydra)', '595_dcn+', '1089_wenet_(penn_treebank)', '633_rnnlm_+_dynamic_kl_regularization_(wt2)', '889_pangu-σ', '1206_emu_(meta)', '677_memsizer', '902_sparse_wide_gpt-3_small', '107_hyperclova', '694_egru_(wt2)', '240_bluumi', '746_llama_guard', '2 1_seamlessm4t'}