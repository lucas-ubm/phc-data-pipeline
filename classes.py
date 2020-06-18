"""The classes used to run the Pharmacogenomic ML pipeline.

This module contains two classes, tuning is used to define a hyper parameter search space to optimize the performance of a method on a validation set. The drug class allows us to create, train and test a drug resistance prediction model for a specific drug. It includes each of the methods necessary for preprocessing, normalization, feature selection, domain adaptation, drug resistance prediction and retrieval of results.

"""

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


class tuning:
    """Class used for defining a hyper parameter search space where a model is optimized.

    This class helps us describe the parameters needed for a Randomized search on a number
    of hyper parameters

    Parameters:
        space (dict): contains the parameters as keys and the possible values for the parameters as values
        iterations (int): the number of iterations to test before determining the optimal hyper parameters
        scoring (scoring method): a scoring function to determine which hyper parameters are best
        cv (int): the number of folds for the cross validation
        jobs (int): the number of cpus used to perform the search
    """
    def __init__(self, space, iterations=100, scoring = 'r2', cv=3, jobs = -2):
        self.space = space
        self.iterations = iterations
        self.scoring = scoring
        self.cv = cv
        self.jobs = jobs

class drug:
    """Class used for training drug resistance model

    The drug class allows us to create, train and test a drug resistance prediction model for a specific drug. It includes each of the methods necessary for preprocessing, normalization, feature selection, domain adaptation, drug resistance prediction and retrieval of results.

    Parameters:
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

        ##These are used to store the
        self.col = []
        self.da = {}
        self.predicted = []

    def norm(self, model):
        """Applies normalization to data

        Retrieves the gene expression data and normalizes it using the method given by `model`. Then it stores the normalized
        data on the gene expression dataframe. For this the method :func:`~methods.norm` is used.

        Args:
            model(`sklearn.base.TransformerMixin`): a normalization method on which fit_transform can be called

        """
        ge = self.ge.copy()

        self.ge = pd.DataFrame(norm(model, self.ge), index=ge.index, columns=ge.keys())

    def pre(self, p = 0.01, t=4):
        """Applies pre-processing to data

        Performs pre-processing on the gene expression data and stores it on the data pandas dataframe.
        To do this it uses the preprocessing method :func:`~methods.pre`

        Args:
            t (float): determines the threshold below which genes are considered to be unexpressed
            p (float): is in the range ]0,1] and determines what is the minimum percentage of the CCLs that
                needs to be expressed. If the actual percentage is smaller then that specific gene will be dropped.
        """
        self.data = pre(self.ge, p, t)



    def combine(self, metric='AUC_IC50'):

        """Combines drug resistance and gene expression data

        This method puts gene expression and drug resistance data together into one dataframe and then stores
        this in the drug's data object. It does so by using the method :func:`~methods.combine`

        Args:
            metric (str): determines the drug resistance measure that will be used.
        """

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
        """Splits the data on train and test set

        This method splits the data into train and test set so methods can be trained on the train set and evaluated on the
        test set. For this sklearn's :func:`~sklearn.model_selection.train_test_split` method is used, stratifying on the domain.
        Then the indices of the elements in the train and test split are stored on two dictionaries so they can be retrieved
        from the drug object.

        Args:
            test (float): determines what percentage of the data is put into the test split. `None` represents the sklearn default value
                which is 0.25
        """

        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(self.metric, axis=1), self.data[self.metric], stratify = self.data.index.get_level_values(0), test_size = test)

        self.X = {'train': X_train.index, 'test': X_test.index}
        self.y = {'train': y_train.index, 'test': y_test.index}


    def get(self, data, split):
        """Returns the data requested

        This method returns the appropriate data that is requested by specifying split and whether labels or features are needed.
        It takes into account if preprocessing, feature selection and domain adaptation have been performed already.

        Args:
            data (str): can be either 'X' or 'y' which will return features or labels respectively.
            split (str): can be either 'train' or 'test' which will return either the train or test split

        Returns:
            A pandas dataframe containing the split of features or labels that was requested
        """

        if data =='y':
            return self.data.loc[self.y[split]][self.metric]
        elif len(self.da) > 0:
            return self.da[split]
        elif len(self.col) > 0:
            return self.data.loc[self.X[split]][self.col].drop(self.metric, axis = 1)
        elif data == 'X':
            return self.data.loc[self.X[split]].drop(self.metric, axis = 1)


    def fs(self, model, n=0, tuning=None):
        """Selects features using the given model

        This method selects the best features based on a given model. Then the optimal features are
        stored so they can be selected to be retrieved when using :func:`~classes.drug.get`. The features
        are selected using :func:`~methods.fs` and then we store the subset of columns selected on
        the `col` variable.

        Args:
            model (estimator object): this is the method that will
                be used to determine the best features
            n (float): determines what percentage of the features will be selected
            tuning (`classes.tuning`): defines a hyper parameter search space over which the feature selection
                method can be optimized.

        """

        X_train, X_test, var = fs(model, self.get('X', 'train'), self.get('X', 'test'), self.get('y', 'train'), n=n, tuning=tuning)
        col = []
        for i, ele in enumerate(var):
            if ele:
                col.append(self.data.keys()[i])
        col.append(self.metric)
        self.col = col

    def ajive(self, joint):
        """Performs Angle-based Joint and Individual Variation (AJIVE), a type of Domain Adaptation

        This method performs domain adaptation on the drug data by using the :func:`~methods.group_ajive` method.
        It then stores the transformed data as a dataframe on the `da` variable.


        Args:
            joint (int): determines the rank of the joint space of the domains
        """
        if len(self.da) == 0:
            data = {'train':self.get('X', 'train'), 'test': self.get('X', 'test')}
            self.da = group_ajive(data, joint)


    def feda(self):
        """Performs Frustratingly Easy Domain Adaptation, a type of Domain Adaptation.

        This method performs domain adaptation on the drug data by using the :func:`~methods.feda` method.
        It then stores the transformed data as a dataframe on the `da` variable.

        """
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
        """Trains model on drug data.

        Trains the model specified on `model` on the train split of the data by using the
        :func:`~methods.drp` method. It then stores the model on the drug's `model` variable.

        Args:
            model (estimator object): model that will be trained on the data
            tuning (`classes.tuning`): hyper parameter search space over which
                the model will be optimized. If not specified default values will
                be used.

        """
        X = self.get('X', 'train')

        self.model = drp(model, X, self.get('y', 'train'), tuning=tuning)

    def predict(self, X = pd.DataFrame()):
        """Predicts the labels of the given samples based on the features using
        the trained model

        Predicts the labels of the samples in `X` using the trained model. In case no `X`
        is provided the test split previously defined will be used. The predicted labels
        are both stored on the drug object and returned.

        Args:
            `X` (DataFrame or numpy array): samples for which the labels should be calculated

        Returns:
            A pandas dataframe or numpy array containing the predicted labels.

        """
        if X.empty:
            X = self.get('X', 'test')

        ypred = self.model.predict(X)
        self.predicted = ypred
        return ypred

    def metrics(self, arr):
        """Calculates the prediction scores as defined on `arr`

        Given the predicted labels and the actual labels it calculates all of the metrics
        defined on `arr`. It then stores the resulting scores on the drug object and returns
        them.

        Args:
            `X` (DataFrame or numpy array): samples for which the labels should be calculated

        Returns:
            A dictionary containing the name of the scoring function as key and the actual
            score as value.

        """
        if len(self.predicted) == 0:
            self.predict()
        scores = {}
        for metric in arr:
            scores[metric.__name__] = metric(self.get('y', 'test'), self.predicted)
        self.scores = scores
        return scores
