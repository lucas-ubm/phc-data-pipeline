from runs import run
from classes import drug
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.svm import SVR
from classes import tuning
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression, SelectFromModel, VarianceThreshold



gdsc_ge = pd.read_csv("/Users/admin/Desktop/Thesis/Code/phc-data-pipeline/data/processed/gdsc_cell_ge.csv", encoding='utf-8').fillna(0).set_index('CCL')
ctrp_ge = pd.read_csv('/Users/admin/Desktop/Thesis/Code/phc-data-pipeline/data/processed/ctrp_cell_ge.csv', encoding='utf-8').fillna(0).set_index('CCL')
gdsc_dr = pd.read_csv('/Users/admin/Desktop/Thesis/Code/phc-data-pipeline/data/processed/gdsc_poz_dr.csv', encoding='utf-8').fillna(0)
ctrp_dr = pd.read_csv('/Users/admin/Desktop/Thesis/Code/phc-data-pipeline/data/processed/ctrp_poz_dr.csv', encoding='utf-8').fillna(0)

ge = {'ctrp': ctrp_ge, 'gdsc': gdsc_ge}
dr = {'ctrp': ctrp_dr, 'gdsc': gdsc_dr}
t2 = {
    'degree': [2, 3, 4, 5],
    'epsilon' : [0.1, 0.2, 0.3, 0.9],
    'C':[0.01, 0.1, 1, 10, 100],
    'gamma':['scale']
}
tuning = tuning(t2, iterations=50, cv=3, scoring='r2')

same = []
for i in gdsc_dr['Drug_name'].unique():
    if i in ctrp_dr['Drug_name'].unique():
        same.append(i)

d1 = str(list(dr.keys()))
d2 = str(same[:10])

fs = 'f_regression'
feda = True
model = 'SVR'
threshold = 0.01
data = 'data'
drugs = d2 

r1, drugs = run(same[:3], ge, dr, f_regression, feda, SVR(), n = threshold, tuning = tuning)

metrics = []
for i in r1.values():
    metrics.append(i.scores)
metrics = str(metrics)
print(metrics)
print('score: '+str(7))