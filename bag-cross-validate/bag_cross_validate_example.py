# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:29:40 2021

@author: vorst
"""

# Python imports

# Third party imports
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, jaccard_score
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import make_scorer

# Local imports
from bag_cross_validate import cross_validate_bag, BagScorer, bags_2_si

# Global definitions

#%%

# Create some dummy data
"""Generate some dummy data
Create bags and single-instance data
A set of bags have a shape [n x (m x p)], and can be through of as an
array of bag instances.
n is the number of bags
m is the number of instances within each bag (this can vary between bags)
p is the feature space of each instance"""
n_bags = 1000
m_instances = 8 # Static number of bags
p = 5
bags = []
# 25% negative class, 75% positive class
# Bags are created with random data, of shape (n, (m,p))
labels = np.concatenate((np.ones(int(n_bags*0.5)),
                         np.zeros(int(n_bags*(1-0.5))),
                         ))
bags = np.random.randint(low=0, high=2, size=(n_bags, m_instances, p))
print("This is what a bag looks like: \n{}".format(bags[0]))

# Split dummy dataset dataset
rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
train_index, test_index = next(rs.split(bags, labels))
train_bags, train_labels = bags[train_index], labels[train_index]
test_bags, test_labels = bags[test_index], labels[test_index]
        
# Create an estimator
dumb = DummyClassifier(strategy='constant', constant=1)
radiusNeighbor = RadiusNeighborsClassifier(weights='distance', 
                                           algorithm='auto',
                                           p=1, # Manhattan distance
                                           )

# Create an evaluation metric
# Multiple evaluation metrics are allowed
accuracy_scorer = make_scorer(accuracy_score)
bagAccScorer = BagScorer(accuracy_scorer) # Accuracy score, no factory function
precision_scorer = make_scorer(precision_score, average='binary')
bagPreScorer = BagScorer(precision_scorer)
jaccard_scorer = make_scorer(jaccard_score, average='binary')
bagJacScorer = BagScorer(jaccard_scorer)
scoring = {'bag_accuracy':bagAccScorer,
           'bag_precision':bagPreScorer,
           'bag_jaccard':bagJacScorer,
           }


#%%

# Cross validate the dummy data and estimator
result_dumb = cross_validate_bag(estimator=dumb, 
                            X=train_bags, 
                            y=train_labels, 
                            groups=None, 
                            scoring=scoring, # Custom scorer... 
                            cv=2,
                            n_jobs=3, 
                            verbose=0, 
                            fit_params=None,
                            pre_dispatch='2*n_jobs', 
                            return_train_score=False,
                            return_estimator=False, 
                            error_score=np.nan)

result_neighbor = cross_validate_bag(estimator=radiusNeighbor, 
                            X=train_bags, 
                            y=train_labels, 
                            groups=None, 
                            scoring=scoring, # Custom scorer... 
                            cv=3,
                            n_jobs=2, 
                            verbose=0, 
                            fit_params=None,
                            pre_dispatch='2*n_jobs', 
                            return_train_score=False,
                            return_estimator=False, 
                            error_score=np.nan)

# Display the results
msg=("\nOur dummy estimator tried his best, and predicted {} percent of bags " 
    "correctly")
msg = msg.format(result_dumb['test_bag_accuracy'])
print(msg)

msg=("\nOur neighbor estimator didnt fair well either, and predicted {} percent "
     "of bags correctly")
msg = msg.format(result_neighbor['test_bag_accuracy'])
print(msg)