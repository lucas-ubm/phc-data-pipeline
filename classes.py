import pandas as pd
import numpy as np
from methods import combine
from methods import pre

class drug:
    
    def __init__(self, name, ge, dr):
        self.name = name
        self.ge = pd.concat(ge, sort = False)
        self.dr = pd.concat(dr, sort = False)
        
    def pre(self, p = 0.01, t=4):
        self.pre = pre(self.ge, p, t)
        
    def combine(self, metric='AUC_IC50'):
        data = []
        for ele in self.ge.index.levels[0]:
            print(ele)
            data.append(combine(self.ge.loc[ele], self.dr.loc[ele], self.name, metric=metric))
        self.data = data
    
        
    def fs(self, model, X = None, y = None, n=0, tuning=None):
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y
            
        self.fs = drp(model, X, y, n, None)
        return self.model
    
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
        
