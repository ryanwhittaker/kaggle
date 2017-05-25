# Basic Linear Regression (Ordinary Least Squares) Model

import os
import csv
import time
import pandas as pd
import statsmodels.formula.api as sm
from lib import readCSV

data_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/'

def train(training_data):
    return sm.ols(formula = 'SalePrice ~ GrLivArea + LotArea + Neighborhood', data = training_data).fit()

def fillNA(data, all_data):
    return data.fillna(all_data.mean())

def run():
    training_data = readCSV(data_path + 'train.csv')
    real_data = readCSV(data_path + 'test.csv')
    model = train(training_data)
    print model.summary()
    real_data['SalePrice'] = model.predict(real_data[['Id', 'GrLivArea', 'LotArea', 'Neighborhood']])
    new_data = real_data[['Id', 'SalePrice']]
    out_path = os.path.dirname(os.path.abspath(__file__)) + '/../results/' + str(int(time.time())) + '.csv'
    new_data.to_csv(out_path, index = False, columns = ['Id', 'SalePrice'])

run()
