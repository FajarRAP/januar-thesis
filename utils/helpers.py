import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def to_percent(number):
    return f"{number * 100:.2f}%"

def min_max_normalization(value, min_value, max_value):
    if min_value == max_value: 
        return 0.0

    return (value - min_value) / (max_value - min_value)

def bulk_min_max_normalization(dataframe: pd.DataFrame):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

def plot_distribution(dataframe: pd.DataFrame):
    columnName = dataframe.columns[0].__str__()
    columnName = columnName.replace("_", " ").title()
    plt.figure(figsize=(6, 4))
    sns.histplot(dataframe.dropna(), kde=True)
    plt.title(f"Distribusi Data {columnName}")
    plt.show()


def missing_value_insight(dataframe: pd.DataFrame):
    missing_values_count = dataframe.isna().sum()
    total_rows = dataframe.shape[0]
    missing_percentage = (missing_values_count / total_rows) * 100
    
    return pd.DataFrame({
        "Missing Values": missing_values_count,
        "Percentage": missing_percentage,
    })
    
def fill_missing_values(dataframe: pd.DataFrame, method: str = 'mean'):
    if method == 'mean':
        fill_value = dataframe.mean()
    elif method == 'median':
        fill_value = dataframe.median()
    elif method == 'mode':
        fill_value = dataframe.mode().iloc[0]
    else:
        raise ValueError("Method must be 'mean', 'median', or 'mode'.")

    return dataframe.fillna(fill_value)