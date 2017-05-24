# Shared Functions For Models

import pandas as pd

def readCSV(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe
