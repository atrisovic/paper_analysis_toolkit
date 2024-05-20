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
        

       
