# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:18:53 2020

@author: z003vrzk
"""

# Python imports
import sys, os

# Third party imports

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

# import bag_cross_validate
from bag_cross_validate import (BagScorer, indexable, check_cv,
                                _check_multimetric_scoring, is_classifier,
                                Parallel, delayed, clone, cross_validate)
from bag_cross_validate import _fit_and_score


#%%

def test_BagScorer():

    from sklearn.metrics import accuracy_score, make_scorer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.model_selection import ShuffleSplit

    """Scoring
    accuracy_scorer must have a _score_func callable with signature
    (y_true, y_pred)
    This is provided by default when using sklearn.metrics.make_scorer
    """
    accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
    bag_metric_scorer = accuracy_scorer
    accuracy_scorer._kwargs
    hasattr(accuracy_scorer, '_score_func')


    # Load data
    from MILCategorization import mil_load
    LoadMIL = mil_load.LoadMIL()
    # Load cat dataset
    _cat_file = r'../data/MIL_cat_dataset.dat'
    _cat_dataset = LoadMIL.load_mil_dataset(_cat_file)
    _cat_bags = _cat_dataset['dataset']
    _cat_bag_labels = _cat_dataset['bag_labels']
    # Split cat dataset
    rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
    train_index, test_index = next(rs.split(_cat_bags, _cat_bag_labels))
    cat_train_bags, cat_train_bag_labels = _cat_bags[train_index], _cat_bag_labels[train_index]
    cat_test_bags, cat_test_bag_labels = _cat_bags[test_index], _cat_bag_labels[test_index]

    # Test estimator
    compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

    # Test custom scorer
    bagScorer = BagScorer(bag_metric_scorer, sparse=True)
    test_score = bagScorer(compNB, cat_train_bags, cat_train_bag_labels)

    return None




def test_fit_and_score():

    from sklearn.metrics import accuracy_score, make_scorer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.model_selection import ShuffleSplit

    # Scoring
    accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
    bag_metric_scorer = accuracy_scorer

    # Load data
    from MILCategorization import mil_load
    LoadMIL = mil_load.LoadMIL()
    # Load cat dataset
    _cat_file = r'../data/MIL_cat_dataset.dat'
    _cat_dataset = LoadMIL.load_mil_dataset(_cat_file)
    _cat_bags = _cat_dataset['dataset']
    _cat_bag_labels = _cat_dataset['bag_labels']
    # Split cat dataset
    rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
    train_index, test_index = next(rs.split(_cat_bags, _cat_bag_labels))
    cat_train_bags, cat_train_bag_labels = _cat_bags[train_index], _cat_bag_labels[train_index]
    cat_test_bags, cat_test_bag_labels = _cat_bags[test_index], _cat_bag_labels[test_index]

    # Test estimator
    compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

    # Test custom scorer
    bagScorer = BagScorer(bag_metric_scorer, sparse=True)
    test_score = bagScorer(compNB, cat_train_bags, cat_train_bag_labels)

    """_fit_and_score testing"""
    X = cat_train_bags
    y = cat_train_bag_labels
    scoring = bagScorer
    estimator = compNB
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

    # Test _fit_and_score
    train, test = next(cv.split(X,y,groups))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    res = _fit_and_score(
            clone(estimator), X, y, scorers, train, test, verbose, parameters,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator,
            error_score=error_score)


    return None




def test_cross_validate():

    from sklearn.metrics import accuracy_score, make_scorer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.model_selection import ShuffleSplit

    # Scoring
    accuracy_scorer = make_scorer(accuracy_score, normalize='weighted')
    bag_metric_scorer = accuracy_scorer

    # Load data
    from MILCategorization import mil_load
    LoadMIL = mil_load.LoadMIL()
    # Load cat dataset
    _cat_file = r'../data/MIL_cat_dataset.dat'
    _cat_dataset = LoadMIL.load_mil_dataset(_cat_file)
    _cat_bags = _cat_dataset['dataset']
    _cat_bag_labels = _cat_dataset['bag_labels']
    # Split cat dataset
    rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
    train_index, test_index = next(rs.split(_cat_bags, _cat_bag_labels))
    cat_train_bags, cat_train_bag_labels = _cat_bags[train_index], _cat_bag_labels[train_index]
    cat_test_bags, cat_test_bag_labels = _cat_bags[test_index], _cat_bag_labels[test_index]

    # Define an estimator
    compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

    # Test custom scorer
    bagScorer = BagScorer(bag_metric_scorer, sparse=True)
    test_score = bagScorer(compNB, cat_train_bags, cat_train_bag_labels)

    # Test cross_validate
    res = cross_validate(compNB, cat_train_bags, cat_train_bag_labels,
                         cv=3, scoring=bagScorer,
                         n_jobs=3, verbose=0, fit_params=None,
                         pre_dispatch='2*n_jobs', return_train_score=False,
                         return_estimator=False, error_score='raise')

    return None