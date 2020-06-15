import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from methods import combine
from methods import pre
from methods import fs
from methods import feda
from methods import drp
from methods import norm
from methods import ajive
from methods import group_ajive
from inspect import getmembers
import time



"""The classes used to run the Pharmacogenomic ML pipeline.

This module contains two classes, tuning is used to define a hyper parameter search space to optimize the performance of a method on a validation set. The drug class allows us to create, train and test a drug resistance prediction model for a specific drug. It includes each of the methods necessary for preprocessing, normalization, feature selection, domain adaptation, drug resistance prediction and retrieval of results.

Example:
  aag = drug('17-AAG', {'ctrp':ctrp_ge, 'gdsc':'gdsc_ge'}, {'ctrp':ctrp_dr, 'gdsc':'gdsc_dr'})
  aag.pre()
  aag.norm(MinMaxScaler)
  aag.combine()
  aag.split()
  aag.fs(ElasticNet, n=0.2, )
  aag.ajive(2)
  aag.train(RandomForestRegressor)
  aag.metrics([r2_score, mean_absolute_error, mean_squared_error, median_absolute_error])
"""


# The tuning class helps define a search space to optimize the hyper parameters of a method using sklearn's
# RandomizedSearchCV
class tuning:
    def __init__(self, space, iterations=100, scoring = 'r2', cv=3, jobs = -2):
        self.space = space
        self.iterations = iterations
        self.scoring = scoring
        self.cv = cv
        self.jobs = jobs        

class drug:
    """Class used for training drug resistance model

    The drug class allows us to create, train and test a drug resistance prediction model for a specific drug. It includes each of the methods necessary for preprocessing, normalization, feature selection, domain adaptation, drug resistance prediction and retrieval of results.
    
    Attributes:
        name (str): the name of the drug for which the model is trained
        ge (dict): a dictionary containing the different domains as keys and the gene expression data
            from that domain as value
        dr (dict): a dictionary containing the different domains as keys and the gene expression data
            from that domain as value
    """
    def __init__(self, name, ge, dr):
        self.name = name
        self.ge = pd.concat(ge, sort = False)
        self.dr = pd.concat(dr, sort = False)
        self.data = pd.DataFrame()
        
        ##Maybe I could have a dict of steps with key = method name value = True if run False else
        self.col = []
        self.da = {}
        self.predicted = []
    
    def norm(self, model):
        ge = self.ge.copy()
        
        self.ge = pd.DataFrame(norm(model, self.ge), index=ge.index, columns=ge.keys())
        
    def pre(self, p = 0.01, t=4):
        self.data = pre(self.ge, p, t)
         
    
        
    def combine(self, metric='AUC_IC50'):
        data = {}
        self.metric = metric
        if not self.data.empty:
            ge = self.data
        else:
            ge = self.ge
        
        for ele in ge.index.levels[0]:
            data[ele] = combine(ge.loc[ele], self.dr.loc[ele], self.name, metric=metric)
        
        del self.dr
        if 'ge' in self.__dict__:
            del self.ge
    
        self.data = pd.concat(data, sort = False).fillna(0)
    
    def split(self, test=None):
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.metric, axis=1), self.data[self.metric], stratify = self.data.index.get_level_values(0), test_size = test)
        
        self.X = {'train': X_train.index, 'test': X_test.index}
        self.y = {'train': y_train.index, 'test': y_test.index}

    
    def get(self, data, split):
        if data =='y':
            return self.data.loc[self.y[split]][self.metric]
        elif len(self.da) > 0:
            return self.da[split]
        elif len(self.col) > 0:
            return self.data.loc[self.X[split]][self.col].drop(self.metric, axis = 1)
        elif data == 'X':
            return self.data.loc[self.X[split]].drop(self.metric, axis = 1)
        
        
    def fs(self, model, n=0, tuning=None):
        X_train, X_test, var = fs(model, self.get('X', 'train'), self.get('X', 'test'), self.get('y', 'train'), n=n, tuning=tuning) 
        col = []
        for i, ele in enumerate(var):
            if ele:
                col.append(self.data.keys()[i])
        col.append(self.metric)
        self.col = col
    
    def ajive(self, joint):
        if len(self.da) == 0:
            data = {'train':self.get('X', 'train'), 'test': self.get('X', 'test')}
            self.da = group_ajive(data, joint)
        
    
    def feda(self):
        train_domains = []
        test_domains = []
        
        X_train = self.get('X', 'train')
        X_test = self.get('X', 'test')
        
        for i in self.data.index.levels[0]:
            if i in self.data.index:
                train_domains.append(X_train.loc[i].to_numpy())
                test_domains.append(X_test.loc[i].to_numpy())
        
            
        self.da['train'] = pd.DataFrame(feda(train_domains))
        self.da['test'] = pd.DataFrame(feda(test_domains))
        
    def train(self, model, tuning=None):
        X = self.get('X', 'train')
        
        self.model = drp(model, X, self.get('y', 'train'), tuning=tuning)
        
    def predict(self, X = pd.DataFrame(), y = pd.DataFrame(), metrics = None):
        if X.empty:
            X = self.get('X', 'test')
        if y.empty:
            y = self.get('y', 'test')
        ypred = self.model.predict(X)
        self.predicted = ypred
        return ypred
    
    def metrics(self, arr: list) -> dict:
        if len(self.predicted) == 0:
            self.predict()
        scores = {}
        for metric in arr:
            scores[metric.__name__] = metric(self.get('y', 'test'), self.predicted)
        self.scores = scores
        return scores

            
        
