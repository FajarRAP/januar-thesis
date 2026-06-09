import pandas as pd

def to_percent(number):
    return f"{number * 100:.2f}%"

def min_max_normalization(dataframe: pd.DataFrame):
  return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())