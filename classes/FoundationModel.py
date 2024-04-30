import json

class FoundationModel:
    def __init__(self, key: str, title: str, id: str, year: int = None):
        self.key = key.lower()
        self.title = title.lower()
        self.id = id
        self.year = year
        
    def as_dict(self):
        return {'modelKey': self.key, 'modelTitle': self.title, 'modelId': self.id, 'modelYear': self.year}
        
        
    def modelsFromJSON(file_path: str):
        with open(file_path, 'r') as f:
            foundational_models_json = json.load(f)
            models = [FoundationModel(
                                    title = data['title'].replace('\\infty', '∞'), 
                                    key = key, 
                                    id = data['paperId'], 
                                    year = data['year']
                        ) 
                            for key, data in foundational_models_json.items()]

        return models