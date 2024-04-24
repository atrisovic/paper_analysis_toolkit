import pandas as pd
from typing import List

# I really don't like how these are organized right now, so this is definitely something with room for improvement.

class Agglomerator:
    def __init__(self):
        pass
    
    def applyQuery(self, df):
        pass
    
class RankedClassificationCounts(Agglomerator):
    def __init__(self, group_key = 'modelKey', additional_key = 'paperId', classification_col = 'classification', ordering_col = 'classification_order'):
        self.group_key = group_key
        self.ordering_col = ordering_col
        self.classification_col = classification_col
        self.additional_key = additional_key
    
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
    
class RankedClassificationCountsYearly(Agglomerator):
    def __init__(self, group_key = 'modelKey', year_key = 'paperYear', additional_key = 'paperId', classification_col = 'classification', ordering_col = 'classification_order'):
        self.group_key = group_key
        self.year_key = year_key
        self.ordering_col = ordering_col
        self.classification_col = classification_col
        self.additional_key = additional_key
    
    def applyQuery(self, df):
        df['classification_ranking'] = df.groupby([self.group_key, self.additional_key])[self.ordering_col].rank(method='min')
        classification_counts = (
            df[df['classification_ranking'] == 1]
                .groupby([self.group_key, self.year_key, self.classification_col])[self.additional_key]
                .nunique()
                .reset_index()
                .rename(columns={self.additional_key:'count'})
                .pivot(index=[self.group_key, self.year_key], columns=self.classification_col, values='count')
                .fillna(0)
            )
        
        return classification_counts
    
    
    
