import pandas as pd
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression, SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

class tuning:

    def __init__(self, space, iterations=100, scoring = 'r2', cv=3, jobs = -2):
        self.space = space
        self.iterations = iterations
        self.scoring = scoring
        self.cv = cv
        self.jobs = jobs 


def fs(model, X, y, n=0, tuning=None):
    # This is used for tree-based feature selection
    if n==0:

        model.fit(X, y)
        if tuning != None:
            r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring)
            r.fit(X, y)
            model = r.best_estimator_

        return SelectFromModel(estimator = model, prefit=True).transform(X)
    
    # This is used for variance threshold selection
    elif n > 1 and type(n) == type(0.2):
        return VarianceThreshold(n).fit_transform(X)

    # This is used for selectkbest and selectpercentile
    else:
        if n < 1:
            n = n*100
            return SelectPercentile(model, n).fit_transform(X, y)
        else:
            return SelectKBest(model, n).fit_transform(X, y)

def drp(model, X, y, tuning=None):
    return model.fit(X, y)
    
    

        

