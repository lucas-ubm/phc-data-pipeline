from runs import run
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


t2 = {
    'degree': [2, 3, 4],
    'epsilon' : [0.1, 0.2, 0.3, 0.9],
    'C':[0.01, 0.1, 1, 10, 1000],
    'gamma':['scale']
}
tuning = tuning(t2, iterations=50, cv=3, scoring='r2')
tuning = None


feda = True
model = 'ElasticNet'
threshold = 0.01
cutoff = 4
test = None

gdsc = True
ctrp = True
ccle = False

fs = 'f_regression'
n = 0.01

data = {'gdsc':gdsc, 'ctrp':ctrp, 'ccle':ccle}




drugs = 3
metric = 'AUC_IC50'

if True in data.values():
    r1, drugs = run(data, fs, feda, model, p = threshold, t=cutoff, tuning = tuning, drugs=drugs, test=test, n=n)
    
    scores = pd.DataFrame.from_dict(drugs, orient='index')
    mean = scores.describe().loc[['mean']]['r2_score'][0]
    std = scores.describe().loc[['std']]['r2_score'][0]
    print('r_2_mean: '+ str(mean))
    print('r_2_std: '+ str(std))
    
    final = {k:v.model.get_params() for k,v in r1.items()}
    
    final = pd.DataFrame.from_dict(final, orient='index')
    scores = scores.join(final)

    
    scores.to_csv('scores.csv')

    scores.boxplot(figsize=(12,9))
    plt.tight_layout()
    plt.savefig('boxplot.png')


