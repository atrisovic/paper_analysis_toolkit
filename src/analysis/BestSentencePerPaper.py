from src.analysis.Agglomerator import Agglomerator

    
class BestSentencePerPaper(Agglomerator):
    def __init__(self, group_key = 'modelId', 
                        year_key = 'paperYear', 
                        additional_key = 'paperId', 
                        classification_col = 'classification', 
                        ordering_col = 'classification_order',
                        resultsfile = None):
        self.group_key = group_key
        self.year_key = year_key
        self.ordering_col = ordering_col
        self.classification_col = classification_col
        self.additional_key = additional_key
        self.resultsfile = resultsfile
    
    def applyQuery(self, df):
        df['classification_ranking'] = df.groupby([self.group_key, self.additional_key])[self.ordering_col].cumcount()
        best_sentences = (df[df['classification_ranking'] == 0])[[self.group_key, self.additional_key, self.classification_col]]
        
        return best_sentences