# Drug resistance learning pipeline

This project aims to develop a pipeline to systematically assess the performance of different ML methods on high dimensional datasets extracted from different domains.

The use case explored in this repository is that of large pharmacogenomic studies, specifically, [GDSC](https://www.cancerrxgene.org/), [CTRP](https://portals.broadinstitute.org/ctrp.v2.1/) and [CCLE](https://portals.broadinstitute.org/ccle).  

First we design a pipeline as shown under

![](graphs/Pipeline.pdf)

This pipeline is documented using docstrings. One can find the documentation as a webpage under [docs/_build/html/index.html](docs/_build/html/index.html)

The pipeline is developed in the files:

* `methods.py`: where we implement the normalization, preprocessing, feature selection, domain adaptation and drug resistance prediction methods. These methods work with [sklearn](https://scikit-learn.org/stable/) methods and could be reused for training any type of tabular data.
* `classes.py`: where we implement the `tuning` and `Drug` classes.
  * `tuning`: is used to define a randomized hyper parameter search for a given drug resistance prediction method.
  * `Drug`: is the central class of the pipeline. It uses the methods defined under `methods.py` and stores the results and data used for modeling a specific drug

* `runs.py`: here we instantiate and run a `Drug` through each of the steps of the pipeline 

* `train.py`: serves as an entry point to the pipeline. Allows to configure a run and store the results. It also defines hyperparameter search spaces for the different drug resistance prediction methods.

* `config.py`: should be created and contain:

  ```python
  dir = '<PATH-TO-PROCESSED-DATA>/data/Processed/'
  guild = '<PATH-TO-GUILD>/venv/.guild/'
  ```

Examples of the use of the pipeline can be found under the jupyter notebooks:

* `methods-example` shows an example of the use of the `methods.py` methods. It trains a model on pharmacogenomic data and displays the results. It can be useful to understand the input given to each of the methods and the output received. By adding `%%time` at the beginning of a cell it could also be used to analyze the time performance of each of the methods. 
* `drug-example`shows an example of the use of the `classes.py` Drug class. Similar to `methods-example` it runs through each of the steps of the pipeline. It can also be used to test performance improvements or the succesful implementation of new methods for the Drug class.
* `run-example` provides an example of the use of the `run` method. 

Two notebooks explore the given pharmacogenomic data:

* `data-cleaning` cleans the name of the CCLs from GDSC, CCLE and CTRP ensuring that the same name format is used for CCLs. Here we also put together CCLs found on Pozdeyev's drug resistance CTRP, GDSC and CCLE data with the ones found on [CellMinderCDB's](https://discover.nci.nih.gov/cellminercdb/) gene expression data.
* `data-exploration` allows us to explore missing CCLs and understanding the intersection between the given datasets.

The last three notebooks are used for analyzing our results:

* `results-append` shows how to add the results of the individual models to the run results data as given by Guild. 
* `analysis-ic-quality` is an analysis of the impact of EC/IC quality on the $r^2$ scores of the models. A threshold is set under which models are disregarded due to the low quality of the data.
* `results-anova` provides an ANOVA analysis of the results based on the different elements of the configuration. Here we also find a statistical analysis of the importance of the different factors.
* `results-hyperparameters` provides an initial exploratory analysis of the impact of the hyperparameters of the three best performing models (Random Forests, K Nearest Neighbours and Elastic Net) on the results.