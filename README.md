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

* `runs.py`: 

