import pandas as pd
import numpy as np


def vector_from_string_bool(vector_string):
    list_elements = filter(None, map(str.strip, vector_string[1:-1].split(' ')))
    mapped_elements = list(map(lambda s: {'true': 1, 'false': 0, '1': 1, '0': 0, '1.':1, '0.': 0}.get(s.lower()), list_elements))
    assert(len(mapped_elements) == 60)
    assert(None not in mapped_elements), vector_string
                           
    return np.array(mapped_elements)

 
def threshold_model(v):
    normalized_v_uses = v[uses_questions].sum()/len(uses_questions)
    uses = normalized_v_uses > uses_threshold
    
    normalized_v_extends = v[extends_questions].sum()/len(extends_questions) 
    extends = normalized_v_extends > extends_threshold
    
    if uses and extends:
        return 'extends'
    
    if uses:
        return 'uses'
    
    return 'context'

results_path = '/home/gridsan/afogelson/osfm/saved_results/classifier/trials/results/premicrosoft_meeting_results.csv'
output_path =  '/home/gridsan/afogelson/osfm/saved_results/classifier/trials/results/premicrosoft_meeting_classified.csv'
vector_column = 'answer_vector'

df = pd.read_csv(results_path)

# these are extracted as highest potentcy from the ones that we asked
uses_questions = [10, 0, 13, 11, 23, 20, 8, 28, 3, 18, 15, 1, 2, 21, 17, 22, 4, 9, 14, 5, 16, 25, 19, 6, 26, 29, 27, 24, 52, 12]
extends_questions = [50, 30, 40, 31, 41, 51, 38, 42, 35, 32, 52, 48, 54, 58, 53, 33, 34, 44, 36, 37, 47, 56, 43, 1, 8, 12, 18, 22, 57, 3]

print(len(set(uses_questions).intersection(set(extends_questions))))
print(len(set(uses_questions).union(set(extends_questions))))


uses_threshold = .3
extends_threshold = .3

df['answer_vector'] = df['answer_vector'].apply(vector_from_string_bool)

df['classification'] = df['answer_vector'].apply(threshold_model)
df.to_csv(output_path, index = False)