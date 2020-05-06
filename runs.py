from classes import drug
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
import config as c
import pandas as pd
#import xgboost as xgb
#import lightgbm as lgb

models = [RandomForestRegressor, SVR, KNeighborsRegressor, RandomForestRegressor, DecisionTreeRegressor, VarianceThreshold, f_regression, mutual_info_regression, linear_model.ElasticNet, linear_model.Lasso] #xgb.XGBRegressor, lgb.LGBMRegressor
models = {k.__name__:k for k in models}

norms = [StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer]
norms = {k.__name__:k for k in norms}

t1 = {
    'epsilon' : [0.1, 0.2, 0.3, 0.9],
    'C':[0.01, 0.1, 1, 10, 1000],
    'kernel' : ['rbf'],
    'gamma':['scale']
}

def run(ge, fs, feda, model, drugs=1000, n=0, ajive = 0,fs_tuning=None, norm='', tuning=None, p = 0.01, t = 4, metric='AUC_IC50', test=None):
    expression_data = {}
    drug_data = {}
    
    # Select datasets
    for k,v in ge.items():
        if v:
            expression_data[k] = pd.read_csv(c.dir+k+'_cell_ge.csv').fillna(0).set_index('CCL')
            drug_data[k] = pd.read_csv(c.dir+k+'_poz_dr.csv').fillna(0)
    

    # Select drugs, first creating a set with all drug names
    names = set([item for sublist in [list(j['Drug_name'].unique()) for j in drug_data.values()] for item in sublist])
    
    for i in drug_data.keys():
        names = names & set(drug_data[i]['Drug_name'].unique())
    names = list(names)
    names = names[:min(len(names), drugs)]
    
    # Select feature selection and model
    fs = models[fs]

    model = models[model]()
    
    # Select normalization
    if norm in norms:
        norm = norms[norm]
    
    drugs = {}
    for i in names:
        ele = drug(i, expression_data, drug_data)
        if not norm=='':
            ele.norm(norm)
        ele.pre(p = p, t = t)
        ele.combine(metric = metric)
        ele.split(test = test)
        if not fs=='':
            ele.fs(fs, n=n, tuning=fs_tuning)
        if feda and ajive ==0:
            ele.feda()
        if ajive > 0 and not feda:
            ele.ajive(ajive)
        ele.train(model, tuning=tuning)
        ele.metrics([r2_score, mean_absolute_error, mean_squared_error, median_absolute_error])
        drugs[i] = ele
        
    scores = {k: v.scores for k,v in drugs.items()}
        
    return drugs, scores

"""
def drug_names(name, dr):
    names
    if name == 'overlap':
        for i in list(dr.values())[0]['Drug_name'].unique():
            if i in list(list(dr.values())[1]['Drug_name'].unique()) and i in  
"""   
   
            
        
    