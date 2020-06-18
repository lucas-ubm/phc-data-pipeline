
rf = {
    'n_estimators' : [10, 50, 100, 150],
    'max_depth' : [2, 3, 4, 5, 6, 7, 8, None],
    'max_features': ['auto', 'sqrt', 'log2'],


}

svr = {
    'C':[1, 10, 50, 100, 250, 500, 750, 1000],
    'kernel':['rbf', 'linear'],
    'gamma':['scale']
}


en = {
    'alpha': [0.5, 1, 1.5, 2, 5],
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'max_iter': [5000]
}

knn = {
    'n_neighbors': range(2, 30, 1),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree'],
    'p': [1,2,3,4],
    'n_jobs': [-1]
}

dt = {
    'criterion': ['mse', 'mae'],
    'max_depth': range(2, 40, 2),
    'max_features': ['auto', 'sqrt', 'log2']
}
