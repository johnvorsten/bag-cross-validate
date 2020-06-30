# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:18:53 2020

@author: z003vrzk
"""

# Python imports
import sys, os

# Third party imports
import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import ShuffleSplit, KFold

# Local imports
from bag_cross_validate import (BagScorer, indexable, check_cv,
                                _check_multimetric_scoring, is_classifier,
                                Parallel, delayed, clone, cross_validate_bag)
from bag_cross_validate import _fit_and_score


#%%

import unittest

class TestBagScorer(unittest.TestCase):

    def test_BagScorer(self):

        """Define scoring functions, such as accuracy or recall,
        which will be used to score how well single-instance inference
        performs on the bag classification task

        The scoring functions have some requirements -
        a) They are passed to BagScorer on initialization
        b) Must have a method "_score_func" with a signature f(y_true, y_pred)
            (This is provided by default when using sklearn.metrics.make_scorer)

        """
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
        bag_metric_scorer = accuracy_scorer
        print(accuracy_scorer._kwargs) # {'normalize':'weighted'}
        hasattr(accuracy_scorer, '_score_func') # True

        """Load data
        Create bags and single-instance data
        A set of bags have a shape [n x (m x p)], and can be through of as an
        array of bag instances.
        n is the number of bags
        m is the number of instances within each bag (this can vary between bags)
        p is the feature space of each instance"""
        n_bags = 100
        m_instances = np.random.randint(10,80,size=n_bags)
        p = 832
        bags = []
        labels = np.random.randint(0, 2, n_bags)
        for m in m_instances:
            _rand = np.random.rand(m, p)
            bag = np.where(_rand < 0.25, 1, 0)
            bags.append(bag)
        bags = np.array(bags)

        # Split data
        rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        train_index, test_index = next(rs.split(bags, labels))
        train_bags, train_labels = bags[train_index], labels[train_index]
        test_bags, test_labels = bags[test_index], labels[test_index]

        """Create a single-instance estimator"""
        compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

        # Test custom scorer
        bagScorer = BagScorer(bag_metric_scorer, sparse=True)
        estimator = bagScorer.estimator_fit(compNB, train_bags, train_labels)
        test_score = bagScorer(estimator, test_bags, test_labels)

        self.assertIsInstance(test_score, float)

    def test_scorer_signature(self):

        """Define scoring functions, such as accuracy or recall,
        which will be used to score how well single-instance inference
        performs on the bag classification task

        The scoring functions have some requirements -
        a) They are passed to BagScorer on initialization
        b) Must have a method "_score_func" with a signature f(y_true, y_pred)
            (This is provided by default when using sklearn.metrics.make_scorer)

        """
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
        print(accuracy_scorer._kwargs) # {'normalize':'weighted'}
        hasattr(accuracy_scorer, '_score_func') # True

        self.assertTrue(hasattr(accuracy_scorer, '_score_func'))

    def test_BagScorer_signature(self):

        # Test custom scorer
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
        bag_metric_scorer = accuracy_scorer
        bagScorer = BagScorer(bag_metric_scorer, sparse=True)

        self.assertTrue('__call__' in dir(bagScorer))

    def test_BagScorer_metric(self):

        """Define scoring functions, such as accuracy or recall,
        which will be used to score how well single-instance inference
        performs on the bag classification task

        The scoring functions have some requirements -
        a) They are passed to BagScorer on initialization
        b) Must have a method "_score_func" with a signature f(y_true, y_pred)
            (This is provided by default when using sklearn.metrics.make_scorer)

        """
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
        bag_metric_scorer = accuracy_scorer
        print(accuracy_scorer._kwargs) # {'normalize':'weighted'}
        hasattr(accuracy_scorer, '_score_func') # True

        """Load data
        Create bags and single-instance data
        A set of bags have a shape [n x (m x p)], and can be through of as an
        array of bag instances.
        n is the number of bags
        m is the number of instances within each bag (this can vary between bags)
        p is the feature space of each instance"""
        n_bags = 100
        m_instances = 5 # Static number of bags
        p = 5
        bags = []
        # 25% negative class, 75% positive class
        labels = np.concatenate((np.ones(int(n_bags*0.75)),
                                 np.zeros(int(n_bags*(1-0.75)))
                                 ))
        for _ in range(n_bags):
            _rand = np.random.rand(m_instances, p)
            bag = np.where(_rand < 0.25, 1, 0)
            bags.append(bag)
        bags = np.array(bags)

        # Split data
        rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        train_index, test_index = next(rs.split(bags, labels))
        train_bags, train_labels = bags[train_index], labels[train_index]
        test_bags, test_labels = bags[test_index], labels[test_index]

        """Create a dummy estimator"""
        dumb = DummyClassifier(strategy='constant', constant=1)
        # concatenate arrays across 1st axis
        SI_train, SI_train_labels = BagScorer.bags_2_si(train_bags, train_labels)
        SI_test, SI_test_labels = BagScorer.bags_2_si(test_bags, test_labels)
        dumb.fit(SI_train, SI_train_labels)
        pred_test = dumb.predict(SI_test)
        pred_train = dumb.predict(SI_train)

        """Calculate the correct number of predictions based on dummy classifier
        The dummy classifier predicts 1 always (constant)
        The training set bas """
        pct_train = sum(train_labels) / len(train_labels)
        pct_test = sum(test_labels) / len(test_labels)
        dumb_accuracy_train = accuracy_score(SI_train_labels, pred_train)
        dumb_accuracy_test = accuracy_score(SI_test_labels, pred_test)

        # Test custom scorer
        bagScorer = BagScorer(bag_metric_scorer, sparse=True)
        estimator = bagScorer.estimator_fit(dumb, train_bags, train_labels)
        test_score = bagScorer(estimator, test_bags, test_labels)
        train_score = bagScorer(estimator, train_bags, train_labels)

        """test_score should output the accuracy for predictions among bags
        The test_score for bagScorer should be equal to the dumb_accuracy_test
        because bag labels are reduced by the most frequest SI prediction

        If all SI labels are predicted + then all bags will be predicted +
        The accuracy of bag labels reduced by BagScorer will be equal to
        percent of bag labels that are positive"""

        self.assertEqual(test_score, pct_test)
        self.assertEqual(train_score, pct_train)
        self.assertEqual(pct_train, dumb_accuracy_train)
        self.assertEqual(pct_test, dumb_accuracy_test)


    def test_cross_validate_bag(self):

        # Scoring
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
        bag_metric_scorer = accuracy_scorer

        """Load data
        Create bags and single-instance data
        A set of bags have a shape [n x (m x p)], and can be through of as an
        array of bag instances.
        n is the number of bags
        m is the number of instances within each bag (this can vary between bags)
        p is the feature space of each instance"""
        n_bags = 100
        m_instances = 5 # Static number of bags
        p = 5
        bags = []
        # 25% negative class, 75% positive class
        labels = np.concatenate((np.ones(int(n_bags*0.5)),
                                 np.zeros(int(n_bags*(1-0.5))),
                                 ))
        for _ in range(n_bags):
            _rand = np.random.rand(m_instances, p)
            bag = np.where(_rand < 0.25, 1, 0)
            bags.append(bag)
        bags = np.array(bags)

        # Split cat dataset
        rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        train_index, test_index = next(rs.split(bags, labels))
        train_bags, train_labels = bags[train_index], labels[train_index]
        test_bags, test_labels = bags[test_index], labels[test_index]

        # Define an estimator
        dumb = DummyClassifier(strategy='constant', constant=1)
        train_accuracy_global = sum(train_labels) / len(train_labels)
        kf = KFold(n_splits = 4)
        accuracies = []
        for train_index, test_index in kf.split(train_labels):
            _fold = train_labels[test_index]
            _acc = sum(_fold) / len(_fold)
            print(sum(_fold))
            accuracies.append(_acc)
        print('Global Accuracy : ', sum(train_labels) / len(train_labels))
        print('Averaged accuracies : ', np.mean(accuracies))

        # Custom scorer
        bagScorer = BagScorer(bag_metric_scorer, sparse=True)

        # Test cross_validate_bag
        res = cross_validate_bag(dumb, train_bags, train_labels,
                             cv=4, scoring=bagScorer,
                             n_jobs=1, verbose=0, fit_params=None,
                             pre_dispatch='2*n_jobs', return_train_score=False,
                             return_estimator=False, error_score='raise')

        """The arithmetic mean of all accuracy predictions should equal the
        prediction accuracy of the training bags (At least if all splits are
        equal size -> Which is not true if the number of training instances
        is not divisible by the number of splits)
        This is only true because the dummy classifier always predicts 1
        If the splits are not equal size then they will be close to equal"""

        self.assertAlmostEqual(np.mean(res['test_score']), train_accuracy_global, 3)


    def test_fit_and_score(self):

        # Scoring
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
        bag_metric_scorer = accuracy_scorer

        """Load data
        Create bags and single-instance data
        A set of bags have a shape [n x (m x p)], and can be through of as an
        array of bag instances.
        n is the number of bags
        m is the number of instances within each bag (this can vary between bags)
        p is the feature space of each instance"""
        n_bags = 100
        m_instances = 5 # Static number of bags
        p = 5
        bags = []
        # 25% negative class, 75% positive class
        labels = np.concatenate((np.ones(int(n_bags*0.5)),
                                 np.zeros(int(n_bags*(1-0.5))),
                                 ))
        for _ in range(n_bags):
            _rand = np.random.rand(m_instances, p)
            bag = np.where(_rand < 0.25, 1, 0)
            bags.append(bag)
        bags = np.array(bags)

        # Split cat dataset
        rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        train_index, test_index = next(rs.split(bags, labels))
        train_bags, train_labels = bags[train_index], labels[train_index]
        test_bags, test_labels = bags[test_index], labels[test_index]

        # Test estimator
        dumb = DummyClassifier(strategy='constant', constant=1)

        # Test custom scorer
        bagScorer = BagScorer(bag_metric_scorer, sparse=True)

        """_fit_and_score testing"""
        X = train_bags
        y = train_labels
        scoring = bagScorer
        estimator = dumb
        groups = None
        cv = 3
        n_jobs=3
        verbose=0
        pre_dispatch=6
        fit_params=None
        return_estimator=None
        error_score='raise'
        return_train_score=None
        parameters=None

        # Test _fit_and_score method
        X, y, groups = indexable(X, y, groups)

        cv = check_cv(cv, y, classifier=is_classifier(estimator))
        scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                            pre_dispatch=pre_dispatch)
        scores = parallel(
            delayed(_fit_and_score)(
                clone(estimator), X, y, scorers, train, test, verbose, parameters,
                fit_params, return_train_score=return_train_score,
                return_times=True, return_estimator=return_estimator,
                error_score=error_score)
            for train, test in cv.split(X, y, groups))









if __name__ == '__main__':
    unittest.main()