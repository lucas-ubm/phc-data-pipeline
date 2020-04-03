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
    
    if fs != None:
        X_train, X_test, var = fs(fs.model, X_train, X_test, fs.tuning)
    
    #if da != None:
        

        
        
    
class tuning:
    def __init__(self, space, iterations=100, scoring = 'r2', cv=3, jobs = -2):
        self.space = space
        self.iterations = iterations
        self.scoring = scoring
        self.cv = cv
        self.jobs = jobs
        

def combine(ge, dr, drug, metric='AUC_IC50'):
        return ge.join(dr[dr['Drug_name'] == drug][['CCL', metric]].set_index('CCL'), how='right')
        
        
def pre(data: pd.DataFrame, p = 0.1, t = 4) -> pd.DataFrame:
    
        under = (data.to_numpy()>t).T.astype(np.int8)

        if p < 1:
            n = [np.count_nonzero(i) > p * data.shape[0] for i in under]
        else:
            n = [np.count_nonzero(i) > p for i in under]
            
        names = {data.keys()[k]:v for k, v in enumerate(n)}
        index = [k for k,v in names.items() if v]
        
        return data[index]



def fs(model, X_train: np.ndarray, X_test: np.ndarray, y:np.ndarray, n=0, tuning=None):
    """ Returns a subset of {X_train} and {X_test} with features being selected by the method {model}
    :param int n: it can be the variance thereshold or the number of chosen features 
    :
    """
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

def jump(domain, n, data):
    result = []
    for i, ele in enumerate(data):
        result.append(ele)
        for j in range(0, domain):
            result.append(0)
        result.append(ele)
        for j in range(0, n-domain-1):
            result.append(0)
    return result
    
def feda(domains):
    n = len(domains)
    
    samples = 0
    for i in domains:
        samples += i.shape[0]
    
    features = domains[0].shape[1]*(n+1)
    
    new = np.zeros(features)
    for i, data in enumerate(domains):
        for j in data:
            new = np.vstack([new, jump(i, n, j)])
            
    return new[1:]

    
    
def drp(model, X, y, tuning=None):
    
    if tuning != None:
        r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring)
        r.fit(X, y)
        drp = r.best_estimator_
    else:
        drp = model.fit(X, y)
    return drp
    
    

        

