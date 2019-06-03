# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from statistics import mean
from deap import creator, base, tools, algorithms
import sys
import warnings

warnings.filterwarnings("ignore")

classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
}

class Utils:
  def __init__(self, df):
    self.df = df
    self.le = LabelEncoder()
    
  def encode(self, class_label):
    self.le.fit(self.df[class_label])
    y = pd.Series(self.le.transform(self.df[class_label]))
    X = self.df.drop([class_label], axis=1)
    return (X, y.values)
   
  def encoder(self):
    return le
  
class Gafes:
  def __init__(self, X, y, n_pop, n_gen):
    self.X = X
    self.y = y
    self.n_pop = n_pop
    self.n_gen = n_gen

  def run(self):
    # get accuracy with all features
    individual = [1 for i in range(len(self.X.columns))]
    print("Accuracy with all features: \t" +
    str(self.get_fitness(individual, self.X, self.y)) + "\n")

    # apply genetic algorithm
    hof = self.genetic_algorithm(self.X, self.y)

    # select the best individual
    accuracy, individual, header = self.get_best_individual(hof, self.X, self.y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))
    print('Feature Subset\t: ' + str(header))

  def get_fitness(self, individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

        # return best score
        return (self.get_best_val_score(X_subset, y),)
    else:
      return(0,)

  def get_best_val_score(self, X, y):
    max_val_score = 0;
    classifier_name = 0;
    for classifier_name, classifier in classifiers.items():
      val_score = mean(cross_val_score(classifier, X, y, cv=5))
      if val_score > max_val_score:
        max_val_score = val_score

    return max_val_score

  def genetic_algorithm(self, X, y):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", self.get_fitness, X=X, y=y)
    toolbox.register("mate", toolas.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=self.n_pop)
    hof = tools.HallOfFame(self.n_pop * self.n_gen)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=self.n_gen, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof

  def get_best_individual(self, hof, X, y):
    """
    Get the best individual
    """
    max_accuracy = 0.0
    for individual in hof:
        if(individual.fitness.values[0] > max_accuracy):
            max_accuracy = individual.fitness.values
            max_individual = individual

    max_individual_header = [list(X)[i] for i in range(len(max_individual))
                               if max_individual[i] == 1]
    return max_individual.fitness.values, max_individual, max_individual_header