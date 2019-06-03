# Gafes

A tool that use Genetic Algorithm for FEature Selection.

## Description

Feature selection is the process of finding the most relevant variables for a predictive model. These techniques can be used to identify and remove unneeded, irrelevant and redundant features that do not contribute or decrease the accuracy of the predictive model. In nature, the genes of organisms tend to evolve over successive generations to better adapt to the environment. The Genetic Algorithm is an heuristic optimization method inspired by that procedures of natural evolution. In feature selection, the function to optimize is the generalization performance of a predictive model. More specifically, we want to minimize the error of the model on an independent data set not used to create the model.

In this project we use [deap](https://deap.readthedocs.io/en/master/) for create the individuals with 'mutations' (subset of columns) and select the best individuals (highest accuracy) in [sklearn](https://scikit-learn.org/) models. You have to read your datase with [pandas](https://pandas.pydata.org/), encode your class labels, create a Gafes object with X, y, number of population (as n_pop) and number of genneration (as n_gen) and run Gafes. In the end, you will have the subset of features that have best accuracy in the population created. 

## Installation

### System requirements

gafes has the following system requirements:
* [Python (>=3.5)](https://www.python.org/downloads/)

### Installing gafes

Please install all dependencies manually with:

```
curl https://raw.githubusercontent.com/anunciado/ICE1047-Gafes/master/requirements.txt | xargs -n 1 -L 1 pip install
```
Then install gafes:

```
!pip install git+https://github.com/anunciado/ICE1047-Gafes.git@master
```
## Example
```python
import pandas as pd
from gafes.gafes import Gafes
from gafes.gafes import Utils

# read dataframe from csv
df = pd.read_csv('dataset.csv')
X, y = Utils(df).encode('class')
gf = Gafes(X=X, y=y, n_pop=20, n_gen=6)
gf.run()
```

See a full example of use in [examples](https://github.com/anunciado/ICE1047-Gafes/tree/master/examples) folder. 

## Authors
### Developers: 
* **Luís Eduardo Anunciado Silva ([cruxiu@ufrn.edu.br](mailto:cruxiu@ufrn.edu.br))** 
### Project Advisor: 
* **Sandro Jose De Souza ([sandro@neuro.ufrn.br](mailto:sandro@neuro.ufrn.br))**

See also the list of [contributors](https://github.com/anunciado/ICE1047-Gafes/contributors) who participated in this project.

## License

This project is licensed under the MIT - see the [LICENSE](LICENSE) file for details

## References

* [Genetic Algorithm For Feature Selection](https://github.com/renatoosousa/GeneticAlgorithmForFeatureSelection) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - A program that search for the best feature subset for you classification mode
* [Genetic Algorithm Feature Selection](https://github.com/scoliann/GeneticAlgorithmFeatureSelection) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - A program that search for the best feature subset for you classification mode
* [Genetic algorithms for feature selection in Data Analytics](https://www.neuraldesigner.com/blog/genetic_algorithms_for_feature_selection) [![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/) - A text that explain the use of genetic algorithms for feature selection
* [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/downloads/breast-cancer-wisconsin-data.zip/2) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) - A data set of breast cancer wisconsin (with diagnostic)
