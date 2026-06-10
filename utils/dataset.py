import pandas as pd

class Dataset:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        # Memisahkan fitur (X) dan target (y)
        self.X = dataframe.iloc[:, :-1]
        self.y = dataframe.iloc[:, -1]
        
        self.X_min = self.X.min()
        self.X_max = self.X.max()

    def split(self, X: pd.DataFrame = None, y: pd.Series = None, test_size: float = .2):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
            
        training_count = int(X.shape[0] * (1 - test_size))
        
        return (X.iloc[:training_count], 
                y.iloc[:training_count], 
                X.iloc[training_count:], 
                y.iloc[training_count:])