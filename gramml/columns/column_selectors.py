import random

    
class FullColumnSelector:
    """
    Selecciona todas las columnas de un tipo determinado
    """
    
    def __init__(self, features):
        self.feature_indexes = features
        
    def select_columns_by_type(self, columns_type):
        return self.feature_indexes.get(columns_type.upper())
    


class RandomColumnSelector: 
    """
    Selecciona un sample de las columnas de un tipo determinado
    """
    
    def __init__(self, features):
        self.feature_indexes = features
        
    def select_columns_by_type(self, columns_type):
        indexes = self.feature_indexes.get(columns_type.upper())
        return random.sample(indexes, random.choice(range(1, len(indexes)))) if len(indexes) > 1 else indexes