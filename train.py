"""This module serves as an entry point for the pipeline where one can either
manually introduce or use tools such as Guild AI to test different configurations.
"""

from runs import run
import numpy as np
from classes import drug
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.svm import SVR
from classes import tuning
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression, SelectFromModel, VarianceThreshold

rf = {
    'n_estimators' : [10, 50, 100, 150],
    'max_depth' : [2, 3, 4, 5, 6, 7, 8, None],
    'max_features': ['auto', 'sqrt', 'log2'],


}

svr = {
    'C':[1, 10, 50, 100, 250, 500, 750, 1000],
    'kernel':['rbf', 'linear'],
    'gamma':['scale']
}


en = {
    'alpha': [0.5, 1, 1.5, 2, 5],
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'max_iter': [5000]
}

knn = {
    'n_neighbors': range(2, 30, 1),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree'],
    'p': [1,2,3,4],
    'n_jobs': [-1]
}

dt = {
    'criterion': ['mse', 'mae'],
    'max_depth': range(2, 40, 2),
    'max_features': ['auto', 'sqrt', 'log2']
}

ts = {'KNeighborsRegressor':knn, 'SVR':svr, 'DecisionTreeRegressor':dt, 'RandomForestRegressor':rf, 'ElasticNet':en}



feda = False
model = 'DecisionTreeRegressor'
threshold = 0.01
cutoff = 4
test = None

ajive = 0
gdsc = False
ctrp = False
ccle = False

fs = 'f_regression'
norm = ''
n = 0.199108

data = {'gdsc':gdsc, 'ctrp':ctrp, 'ccle':ccle}
tuning = tuning(ts[model], iterations=35, cv=3, scoring='r2', jobs = -1)


drugs = 1
metric = 'AUC_EC50'

# Make sure at least one dataset was selected
if True in data.values():

    r1, drugs = run(data, fs, feda, model, p = threshold, t=cutoff, tuning = tuning, drugs=drugs, test=test, n=n, norm=norm)

    scores = pd.DataFrame.from_dict(drugs, orient='index')
    mean = scores.describe().loc[['mean']]['r2_score'][0]
    std = scores.describe().loc[['std']]['r2_score'][0]
    print('r_2_mean: '+ str(mean))
    print('r_2_std: '+ str(std))

    result = {k:v.model.get_params() for k,v in r1.items()}

    scores = scores.join(pd.DataFrame.from_dict(result, orient='index'))

    scores.to_csv('scores.csv')


    scores.boxplot(figsize=(12,9))
    plt.tight_layout()
    plt.savefig('boxplot.png')
