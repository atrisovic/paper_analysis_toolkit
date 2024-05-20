from src.analysis.Agglomerator import Agglomerator

class RankedClassificationCounts(Agglomerator):
    def __init__(self, group_key = 'modelId', 
                        additional_key = 'paperId', 
                        classification_col = 'classification', 
                        ordering_col = 'classification_order',
                        resultsfile = None):
        self.group_key = group_key
        self.ordering_col = ordering_col
        self.classification_col = classification_col
        self.additional_key = additional_key
        self.resultsfile = resultsfile
    
    def applyQuery(self, df):
        df['classification_ranking'] = df.groupby([self.group_key, self.additional_key])[self.ordering_col].rank(method='min')
        classification_counts = (df[df['classification_ranking'] == 1]
                                                .groupby([self.group_key, self.classification_col])[self.additional_key]
                                                .nunique()
                                                .reset_index()
                                                .rename(columns={self.additional_key:'count'})
                                                .pivot(index=self.group_key, columns=self.classification_col, values='count')
                                                .fillna(0)
                                                .rename_axis(None, axis = 1)
                                                .reset_index()
                                                .set_index(self.group_key)
                                            )
        return classification_counts