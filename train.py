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
import xgboost as xgb
from classes import tuning
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression, SelectFromModel, VarianceThreshold
from hyperopt import hp

t1 = {
    'n_estimators' : [10, 50, 100, 150],
    'max_depth' : [2, 3, 4, 5, 6, 7, 8, None],
    'max_features': ['auto', 'sqrt', 'log2'],
    
    
}

t2 = {
    'C':[1, 10, 50, 100, 250, 500, 750, 1000],
    'kernel':['rbf', 'linear'],
    'gamma':['scale']
}


t3 = {
    'alpha': [0.5, 1, 1.5, 2, 5],
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'max_iter': [5000]
}

t4 = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5, 6]
}

t5 = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'num_leaves': range(30,150,10),
    'min_child_samples': range(20,500,20),
    'reg_alpha': np.linspace(0, 1, num=10),
    'reg_lambda': np.linspace(0, 1, 10)
    
}

t6 = {
    'n_neighbors': range(2, 30, 1),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree'],
    'p': [1,2,3,4],
    'n_jobs': [-1]
}

t7 = {
    'criterion': ['mse', 'mae'],
    'max_depth': range(2, 40, 2),
    'max_features': ['auto', 'sqrt', 'log2']
}

tuning = tuning(t7, iterations=50, cv=5, scoring='r2', jobs = -1)



feda = False
model = 'DecisionTreeRegressor'
threshold = 0.01
cutoff = 4
test = None

gdsc = True
ctrp = True
ccle = False

fs = 'Lasso'
n = 0.005

data = {'gdsc':gdsc, 'ctrp':ctrp, 'ccle':ccle}




drugs = 5
metric = 'AUC_IC50'

if True in data.values():
    r1, drugs = run(data, fs, feda, model, p = threshold, t=cutoff, tuning = tuning, drugs=drugs, test=test, n=n)
    
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


