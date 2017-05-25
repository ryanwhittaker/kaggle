# Basic Linear Regression (Ordinary Least Squares) Model

import os
import csv
import time
import pandas as pd
import statsmodels.formula.api as sm
from lib import readCSV

data_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/'
model_vars = ['GrLivArea', 'LotArea', 'Neighborhood', 'OverallQual', 'OverallCond', 'MSSubClass']

def train(training_data):
    formula = 'SalePrice ~ ' + ' + '.join(model_vars)
    print formula
    return sm.ols(formula = formula, data = training_data).fit()

def categorizeData(data):
    standard_category_fields = ['Neighborhood', 'OverallQual', 'OverallCond']
    for field in standard_category_fields:
        data[field] = data[field].astype('category')
    data['MSSubClass'] = data['MSSubClass'].astype('category', categories = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190])
    return data

def fillNA(data, all_data):
    data = data.fillna(all_data.mean())
    return data

def run():
    # Read in the Data
    training_data = readCSV(data_path + 'train.csv')
    real_data = readCSV(data_path + 'test.csv')

    # Concat data together
    all_data = training_data.append(real_data, ignore_index = True)

    # Fill in gaps
    training_data = fillNA(training_data, all_data)
    real_data = fillNA(real_data, all_data)

    # Categorize fields
    training_data = categorizeData(training_data)
    real_data = categorizeData(real_data)

    # Train the model and print the summary
    model = train(training_data)
    print model.summary()

    # Fill model and gather predictions. Store in SalePrice
    model_vars.append('Id')
    real_data['SalePrice'] = model.predict(real_data[model_vars])

    # Strip other fields down
    new_data = real_data[['Id', 'SalePrice']]

    # Output to results
    out_path = os.path.dirname(os.path.abspath(__file__)) + '/../results/' + str(int(time.time())) + '.csv'
    new_data.to_csv(out_path, index = False, columns = ['Id', 'SalePrice'])

run()
