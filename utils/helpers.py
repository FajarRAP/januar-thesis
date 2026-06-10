import pandas as pd

def to_percent(number):
    return f"{number * 100:.2f}%"

def min_max_normalization(value, min_value, max_value):
    if min_value == max_value: 
        return 0.0

    return (value - min_value) / (max_value - min_value)