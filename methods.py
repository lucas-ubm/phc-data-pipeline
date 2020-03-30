import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression, SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

class drug:
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y
        
    def select(self, model, X = 0, y = 0, n=0, tuning=None):
        return self
    
    def train(self, model):
        return self
        
    
class tuning:
    def __init__(self, space, iterations=100, scoring = 'r2', cv=3, jobs = -2):
        self.space = space
        self.iterations = iterations
        self.scoring = scoring
        self.cv = cv
        self.jobs = jobs 
        
    
def pre(data: pd.DataFrame,p = 0.1, t = 4) -> pd.DataFrame:
        
        under = data.applymap(lambda x: np.nan if (x<=t) else x)
        
        if p < 1:
            under = under.count() > p * data.shape[0]
        else:
            under = under.count() > p
        
        index = [k for k,v in under.items() if v]        
        return data[index]
    
def combine(ge, dr, drug):
        drug_dr = dr[dr['Drug_name'] == drug][['CCL', 'AUC_IC50']]
        
        X = np.array([list(ge.loc[i].values) for i in drug_dr[drug_dr['CCL'].isin(ge.index)]['CCL']])
        X = X.reshape(drug_dr.shape[0], ge.shape[1])
        y = drug_dr['AUC_IC50'].to_numpy()
        
        return X, y

def fs(model, X: np.ndarray, y:np.ndarray, n=0, tuning=None):
    # This is used for tree-based feature selection
    if n==0:

        model.fit(X, y)
        if tuning != None:
            r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring)
            r.fit(X, y)
            model = r.best_estimator_
        fs = SelectFromModel(estimator = model, prefit=True)
        return fs.transform(), fs.get_support()
    
    # This is used for variance threshold selection
    elif n > 1 and type(n) == type(0.2):
        fs = VarianceThreshold(n)
        return fs.fit_transform(X), fs.get_support()

    # This is used for selectkbest and selectpercentile
    else:
        if n < 1:
            n = n*100
            fs = SelectPercentile(model, n)
            return fs.fit_transform(X, y), fs.get_support()
        else:
            fs = SelectKBest(model, n)
            return fs.fit_transform(X, y), fs.get_support()

def drp(model, X, y, tuning=None):
    
    if tuning != None:
        r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring)
        r.fit(X, y)
        drp = r.best_estimator_
    else:
        drp = model.fit(X, y)
    return drp
    
    

        

