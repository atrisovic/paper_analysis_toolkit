import logging

# I really don't like how these are organized right now, so this is definitely something with room for improvement.

logger = logging.getLogger(__name__)


class Agglomerator:
    def __init__(self, resultsfile: str = None):
        self.resultsfile = resultsfile
    
    def applyQuery(self, df):
        pass
    
    def saveQuery(self, df, info: str = None):
        if (self.resultsfile is None):
            return
        
        results = self.applyQuery(df)
        
        if results is not None:
            with open(self.resultsfile, 'a+') as f:
                f.write(results.to_json(orient = 'index') + "\n")
                logger.info(f"Saved results with info string: {info}.")
        else:
            logger.info(f"Empty results found for info: {info}, nothing saved.")
        
class RankedClassificationCounts(Agglomerator):
    def __init__(self, group_key = 'modelKey', 
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
       
class RankedClassificationCountsYearly(Agglomerator):
    def __init__(self, group_key = 'modelKey', 
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
    
class BestSentencePerPaper(Agglomerator):
    def __init__(self, group_key = 'modelKey', 
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