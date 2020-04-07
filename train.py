from runs import run
from classes import drug
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.svm import SVR
from classes import tuning
import pandas as pd

gdsc_ge = pd.read_csv('data/Processed/gdsc_cell_ge.csv').fillna(0).set_index('CCL')
ctrp_ge = pd.read_csv('data/Processed/ctrp_cell_ge.csv').fillna(0).set_index('CCL')
gdsc_dr = pd.read_csv('data/Processed/gdsc_poz_dr.csv').fillna(0)
ctrp_dr = pd.read_csv('data/Processed/ctrp_poz_dr.csv').fillna(0)

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
data = d1
drugs = d2 

r1 = run(same[:10], ge, dr, f_regression, feda, SVR(), n = threshold, tuning = tuning)

metrics = []
for i in r1:
    metrics.append(r1.scores)