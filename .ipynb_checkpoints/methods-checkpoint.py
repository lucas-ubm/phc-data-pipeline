import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression, SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


def run(drug, pre, norm, fs, da, model, test = None):
    X = drug.get_x()
    y = drug.get_y()
    
    if pre != None:
        X = pre(X, pre)
    if norm != None:
        X, y = norm(X, y, norm)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test)
    #if fs != None:
        
        

class drug:
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y
        
    def select(self, model, X = None, y = None, n=0, tuning=None):
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y
            
        self.model = drp(model, X, y, n, None)
    
    def train(self, model, X = None, y = None, tuning=None):
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y
            
        self.model = drp(model, X, y, None)
        
        
    
class tuning:
    def __init__(self, space, iterations=100, scoring = 'r2', cv=3, jobs = -2):
        self.space = space
        self.iterations = iterations
        self.scoring = scoring
        self.cv = cv
        self.jobs = jobs 
        

def combine(ge, dr, drug):
        drug_dr = dr[dr['Drug_name'] == drug][['CCL', 'AUC_IC50']]
        
        X = np.array([list(ge.loc[i].values) for i in drug_dr[drug_dr['CCL'].isin(ge.index)]['CCL']])
        X = X.reshape(drug_dr.shape[0], ge.shape[1])
        y = drug_dr['AUC_IC50'].to_numpy()
        
        return X, y   
        
        
def pre(data: pd.DataFrame,p = 0.1, t = 4) -> pd.DataFrame:
        
        under = data.applymap(lambda x: np.nan if (x<=t) else x)
        
        if p < 1:
            under = under.count() > p * data.shape[0]
        else:
            under = under.count() > p
        
        index = [k for k,v in under.items() if v]        
        return data[index]
    



def fs(model, X_train: np.ndarray, X_test: np.ndarray, y:np.ndarray, n=0, tuning=None):
    # This is used for tree-based feature selection
    if n==0:

        model.fit(X_train, y)
        
        if tuning != None:
            r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring)
            r.fit(X_train, y)
            model = r.best_estimator_
            
        fs = SelectFromModel(estimator = model, prefit=True)
        X_train = fs.transform()
    
    # This is used for variance threshold selection
    elif n > 1 and type(n) == type(0.2):
        fs = VarianceThreshold(n)
        X_train = fs.fit_transform(X_train)

    # This is used for selectkbest and selectpercentile
    else:
        if n < 1:
            n = n*100
            fs = SelectPercentile(model, n)
        else:
            fs = SelectKBest(model, n)
            
        X_train = fs.fit_transform(X_train, y)
        
    var = fs.get_support()
    index = [i for i, x in enumerate(var) if x]
    X_test = np.apply_along_axis(lambda x: x[index], 1, X_test)
    
    return X_train, X_test, var

#def select()
        
def drp(model, X, y, tuning=None):
    
    if tuning != None:
        r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring)
        r.fit(X, y)
        drp = r.best_estimator_
    else:
        drp = model.fit(X, y)
    return drp
    
    

        

