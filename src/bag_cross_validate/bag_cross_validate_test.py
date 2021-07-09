# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:18:53 2020

@author: z003vrzk
"""

# Python imports
import sys, os
import unittest

# Sklearn imports
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.base import is_classifier, clone
from sklearn.metrics._scorer import check_scoring
from sklearn.model_selection._split import check_cv
from sklearn.utils import indexable
from sklearn.neighbors import KNeighborsClassifier

# Third party imports
from joblib import Parallel, delayed
import numpy as np
from scipy.sparse import csr_matrix

# Local imports
from bag_cross_validate import (BagScorer, 
                                cross_validate_bag,
                                _fit_and_score,
                                bags_2_si,)


#%%

class TestBagScorer(unittest.TestCase):
    
    def setUp(self):
        
        """Generate some dummy data
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
        # Bags are created with random data, of shape (n, (m,p))
        labels = np.concatenate((np.ones(int(n_bags*0.5)),
                                 np.zeros(int(n_bags*(1-0.5))),
                                 ))
        for _ in range(n_bags):
            _rand = np.random.rand(m_instances, p)
            bag = np.where(_rand < 0.25, 1, 0)
            bags.append(bag)
        bags = np.array(bags)

        # Split dummy dataset dataset
        rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        train_index, test_index = next(rs.split(bags, labels))
        train_bags, train_labels = bags[train_index], labels[train_index]
        test_bags, test_labels = bags[test_index], labels[test_index]
        self.train_bags, self.train_labels = bags[train_index], labels[train_index]
        self.test_bags, self.test_labels = bags[test_index], labels[test_index]
        
        return None


    def test_BagScorer(self):

        """Define scoring functions, such as accuracy or recall,
        which will be used to score how well single-instance inference
        performs on the bag classification task

        The scoring functions have some requirements -
        a) They are passed to BagScorer on initialization
        b) Must have a method "_score_func" with a signature f(y_true, y_pred)
            (This is provided by default when using sklearn.metrics.make_scorer)

        """
        
        # Create scoring metrics, and load scoring metric into BagScorer
        accuracy_scorer = make_scorer(accuracy_score, normalize=True)
        precision_scorer = make_scorer(precision_score, average='weighted')
        recall_scorer = make_scorer(recall_score, average='weighted')
        # {'normalize':'weighted'}
        self.assertDictContainsSubset({'normalize':True}, accuracy_scorer._kwargs)
        self.assertIn('_score_func', accuracy_scorer.__dict__.keys())

        # Dummy data
        train_bags, train_labels = self.train_bags, self.train_labels
        test_bags, test_labels = self.test_bags, self.test_labels

        # Create a single-instance estimator
        compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

        # Test custom scorer
        bagAccScorer = BagScorer(accuracy_scorer, sparse=True)
        bagPrecisionScorer = BagScorer(precision_scorer, sparse=True)
        bagRecallScorer = BagScorer(recall_scorer, sparse=True)
        estimator = bagAccScorer.estimator_fit(compNB, train_bags, train_labels)
        
        # The estimator is the same for all instances...
        accuracy = bagAccScorer(estimator, test_bags, test_labels)
        precision = bagPrecisionScorer(estimator, test_bags, test_labels)
        recall = bagRecallScorer(estimator, test_bags, test_labels)

        self.assertIsInstance(accuracy, float)
        self.assertLess(accuracy, 1)
        self.assertGreater(accuracy, 0)
        
        self.assertIsInstance(precision, float)
        self.assertLess(precision, 1)
        self.assertGreater(precision, 0)
        
        self.assertIsInstance(recall, float)
        self.assertLess(recall, 1)
        self.assertGreater(recall, 0)
        
        return None
    

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
        bagAccScorer = BagScorer(accuracy_scorer, sparse=True)

        self.assertTrue(callable(bagAccScorer), msg="BagScorer must be callable")
        
        return None


    def test_BagScorer_metric(self):

        """Define scoring functions, such as accuracy or recall,
        which will be used to score how well single-instance inference
        performs on the bag classification task

        The scoring functions have some requirements -
        a) They are passed to BagScorer on initialization
        b) Must have a method "_score_func" with a signature f(y_true, y_pred)
            (This is provided by default when using sklearn.metrics.make_scorer)

        Successful conditions:
            The bagscorer must report the same performance metrics as when the 
            metrics are manually calculated
        This tests if the bagscorer property fits, trains, and evaluates
        the estimator passed to it
        """
        
        # Generate a scoring metric for the bag scorer
        accuracy_scorer = make_scorer(accuracy_score)
        self.assertTrue(hasattr(accuracy_scorer, '_score_func'), 
                        msg='accuracy scorer must have _score_function method') 

        # Generate some data
        train_bags, train_labels = self.train_bags, self.train_labels
        test_bags, test_labels = self.test_bags, self.test_labels

        # Create a dummy estimator
        dumb = DummyClassifier(strategy='constant', constant=1)
        
        # concatenate arrays across 1st axis
        SI_train, SI_train_labels = bags_2_si(train_bags, train_labels)
        SI_test, SI_test_labels = bags_2_si(test_bags, test_labels)
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

        # Test custom scorer, with the same dummy estimator
        bagAccScorer = BagScorer(accuracy_scorer, sparse=True)
        estimator = bagAccScorer.estimator_fit(dumb, train_bags, train_labels)
        test_score = bagAccScorer(estimator, test_bags, test_labels)
        train_score = bagAccScorer(estimator, train_bags, train_labels)

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

        # Dummy data
        train_bags, train_labels = self.train_bags, self.train_labels
        test_bags, test_labels = self.test_bags, self.test_labels

        # Define an estimator
        dumb = DummyClassifier(strategy='constant', constant=1)
        
        # Calculate metrics manually
        expected_accuracy = sum(train_labels) / len(train_labels)
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
        bagAccScorer = BagScorer(accuracy_scorer, sparse=True)
        scorer = {'bag-accuracy-scorer': bagAccScorer,
                   }

        # Test cross_validate_bag
        # Res is a dictonary of lists {'fit_time':[1,2,3],
        # 'test_bag-accuracy-scorer':[0.1,0.2,0.3]}
        res = cross_validate_bag(dumb, train_bags, train_labels,
                             cv=4, scoring=scorer,
                             n_jobs=1, verbose=0, fit_params=None,
                             pre_dispatch='2*n_jobs', return_train_score=False,
                             return_estimator=False, error_score='raise')

        """The arithmetic mean of all accuracy predictions should equal the
        prediction accuracy of the training bags (At least if all splits are
        equal size -> Which is not true if the number of training instances
        is not divisible by the number of splits)
        This is only true because the dummy classifier always predicts 1
        If the splits are not equal size then they will be close to equal"""
        self.assertAlmostEqual(np.mean(res['test_bag-accuracy-scorer']), 
                               expected_accuracy, 3)
        # Just check the mean also LOL
        self.assertEqual(np.mean(res['test_bag-accuracy-scorer']), 
                         expected_accuracy)
        # 4 Crossvalidation splits
        self.assertTrue(len(res['test_bag-accuracy-scorer']) == 4)
        # Assert result has dictionary values
        self.assertIn('fit_time', res.keys())
        self.assertIn('score_time', res.keys())
        
        return None


    def test_fit_and_score(self):

        # Scoring
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')

        # Test estimator
        dumb = DummyClassifier(strategy='constant', constant=1)

        # Test custom scorer
        bagAccScorer = BagScorer(accuracy_scorer, sparse=True)

        # _fit_and_score testing
        X = self.train_bags
        y = self.train_labels
        scoring = {'bag-accuracy-scorer': bagAccScorer,
                   }
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
        scorers = _check_multimetric_scoring(estimator, scoring=scoring)

        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                            pre_dispatch=pre_dispatch)
        
        # Scores is a list of dictonaries
        """When scoring is a dictionary, the returned result looks like
        [{'test_scores': {'bag-accuracy-scorer': 0.5185185185185185},
          'fit_time': 0.0,
          'score_time': 0.0},
         {'test_scores': {'bag-accuracy-scorer': 0.5185185185185185},
          'fit_time': 0.0,
          'score_time': 0.0}, ... ]"""
        scores = parallel(
            delayed(_fit_and_score)(
                clone(estimator), X, y, scorers, train, test, verbose, parameters,
                fit_params, return_train_score=return_train_score,
                return_times=True, return_estimator=return_estimator,
                error_score=error_score)
            for train, test in cv.split(X, y, groups))
        
        for score in scores:
            bag_scoring_metric = score['test_scores']
            self.assertLessEqual(bag_scoring_metric['bag-accuracy-scorer'], 1)
            self.assertGreaterEqual(bag_scoring_metric['bag-accuracy-scorer'], 0)
            
            fit_time = score['fit_time']
            self.assertIsInstance(fit_time, float)
            
            score_time = score['score_time']
            self.assertIsInstance(score_time, float)
        
        return None
    
    
    def test_fit_and_score_return_dict(self):
        
        # Scoring
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
        
        # Test estimator
        dumb = DummyClassifier(strategy='constant', constant=1)

        # Test custom scorer
        bagAccScorer = BagScorer(accuracy_scorer, sparse=True)
        
        # Rename for easier parameters
        X = self.train_bags
        y = self.train_labels
        scoring = {'bag-scorer':bagAccScorer}
        estimator = dumb
        groups = None
        cv = 3
        n_jobs=3
        verbose=0
        pre_dispatch=6
        fit_params=None
        return_estimator=True
        error_score='raise'
        return_train_score=True
        parameters=None
        
        # Test _fit_and_score method
        X, y, groups = indexable(X, y, groups)
        cv = check_cv(cv, y, classifier=is_classifier(estimator))
        scorers = _check_multimetric_scoring(estimator, scoring=scoring)

        # Use one cross-validation split
        generator = cv.split(X, y, groups)
        # Get training and test split of training data
        train, test = next(generator)
        # Generate scores using BagScorer
        scores = _fit_and_score(
            clone(estimator), X, y, scorers, train, test, verbose, parameters,
            fit_params, 
            return_train_score=return_train_score,
            return_times=True, 
            return_estimator=return_estimator,
            return_n_test_samples=False,
            error_score=error_score)
        
        # Returned dictionary contains keys
        self.assertIn('train_scores', scores.keys())
        self.assertIn('test_scores', scores.keys())
        self.assertIn('fit_time', scores.keys())
        self.assertIn('score_time', scores.keys())
        self.assertIn('estimator', scores.keys())
        
        return None



class TestCrossValidation(unittest.TestCase):
    
    def setUp(self):
        """Generate some dummy data
        Create bags and single-instance data
        A set of bags have a shape [n x (m x p)], and can be through of as an
        array of bag instances.
        n is the number of bags
        m is the number of instances within each bag (this can vary between bags)
        p is the feature space of each instance"""
        n_bags = 100
        m_instances_range = [5,10] # Dynamic number instance per bag
        p = 5
        bags_sparse = []
        # 25% negative class, 75% positive class
        # Bags are created with random data, of shape (n, (m,p))
        labels = np.concatenate((np.ones(int(n_bags*0.5)),
                                 np.zeros(int(n_bags*(1-0.5))),
                                 ))
        for n in range(0, n_bags):
            m_instances = np.random.randint(m_instances_range[0], 
                                            m_instances_range[1],)
            bag = np.random.randint(low=0, high=2, size=(m_instances, p))
            bag_sparse = csr_matrix(bag)
            bags_sparse.append(bag_sparse)
            
        bags_sparse = np.array(bags_sparse)

        # Split dummy dataset dataset
        rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        train_index, test_index = next(rs.split(bags_sparse, labels))
        train_bags, train_labels = bags_sparse[train_index], labels[train_index]
        test_bags, test_labels = bags_sparse[test_index], labels[test_index]
        self.train_bags, self.train_labels = bags_sparse[train_index], labels[train_index]
        self.test_bags, self.test_labels = bags_sparse[test_index], labels[test_index]
        
        return None
    
    
    def test_bag_cross_validate_sparse(self):
        
        # Scoring
        accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')

        # Dummy data
        train_bags, train_labels = self.train_bags, self.train_labels
        test_bags, test_labels = self.test_bags, self.test_labels

        # Define an estimator
        estimator = KNeighborsClassifier(n_neighbors=10, 
                                         weights='uniform',
                                         algorithm='ball_tree', 
                                         n_jobs=4)

        # Custom scorer
        bagAccScorer = BagScorer(accuracy_scorer, input_sparse=True)
        scorer = {'bag-accuracy-scorer': bagAccScorer,
                  }

        # Test cross_validate_bag
        # Res is a dictonary of lists {'fit_time':[1,2,3],
        # 'test_bag-accuracy-scorer':[0.1,0.2,0.3]}
        res = cross_validate_bag(estimator, train_bags, train_labels,
                                 cv=4, scoring=scorer,
                                 n_jobs=1, verbose=0, fit_params=None,
                                 pre_dispatch='2*n_jobs', return_train_score=False,
                                 return_estimator=False, error_score='raise')


        # 4 Crossvalidation splits
        self.assertTrue(len(res['test_bag-accuracy-scorer']) == 4)
        # Assert result has dictionary values
        self.assertIn('fit_time', res.keys())
        self.assertIn('score_time', res.keys())
        
        return None












#%%

if __name__ == '__main__':
    # Run all test cases
    unittest.main()
    
    # # Run specific test methods...
    # runner = unittest.TextTestRunner()
    # classes = [TestBagScorer]
    # unit_tests_to_run = [
    #     'test_BagScorer',
    #     'test_BagScorer_dict',
    #     'test_BagScorer_metric',
    #     'test_BagScorer_signature',
    #     'test_cross_validate_bag',
    #     'test_fit_and_score',
    #     'test_fit_and_score_return_dict',
    #     'test_scorer_signature',
    #     ]
    
    # Run specific test methods... (altenative method)
    # suite = unittest.TestSuite()
    # suite.addTest(TestBagScorer('test_fit_and_score_return_dict'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)