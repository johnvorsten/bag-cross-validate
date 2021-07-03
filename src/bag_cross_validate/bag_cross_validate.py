# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 20:53:13 2020

@author: z003vrzk
"""


# Python imports
from collections import Counter
import warnings
import numbers
import time
from traceback import format_exc
from typing import Union

# Sklearn imports
from sklearn.utils.validation import (_check_fit_params, 
                                      check_is_fitted,
                                      NotFittedError, 
                                      _num_samples,
                                      _deprecate_positional_args)
from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection._validation import (_score, 
                                                 _aggregate_score_dicts, 
                                                 logger)
from sklearn.base import (is_classifier, 
                          clone)
from sklearn.utils import indexable
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv

# Third party imports
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from joblib import (Parallel, 
                    delayed)
import numpy as np

#%%

def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_estimator=False,
                   split_progress=None, candidate_progress=None,
                   error_score=np.nan):

    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    split_progress : list or tuple, optional, default: None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>)
    candidate_progress : list or tuple, optional, default: None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>)
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
    """
    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += (f"; {candidate_progress[0]+1}/"
                             f"{candidate_progress[1]}")

    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = (', '.join(f'{k}={parameters[k]}'
                                    for k in sorted_keys))
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # Fit the estimator
    # Let the custom bag scorer handle fitting of the estimator
    result = {}
    try:
        estimator = _BagScorer_estimator_fit(estimator, 
                                             X_train, 
                                             y_train, 
                                             scorer, 
                                             **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")
    
    
    else:
        # The estimator is fitted to data correctly
        # Calculate scoring of estimator using custom scorer
        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        result_msg += (f"train="
                                       f"{train_scores[scorer_name]:.3f}, ")
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        # Return the number of bags
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
        
    return result


def _BagScorer_estimator_fit(estimator, X_train, y_train, scorer, **fit_params):
    
    generic_error = ("Invalid scorer object. Should be BagScorer or dictionary"
                     " with structure {'key':BagScorer} object. Got {scorer!r}")
    
    if y_train is None:
        # X_train is an iterabls of bags NOT single-instance examples
        if isinstance(scorer, BagScorer):
            # Depreciated
            estimator = scorer.estimator_fit(estimator, 
                                             X_train,
                                             y_train=None, 
                                             **fit_params)
        elif isinstance(scorer, dict):
            # It is possible to pass a dictionary of scorers
            for metric, bagScorer in scorer.items():
                if isinstance(bagScorer, BagScorer):
                    estimator = bagScorer.estimator_fit(estimator, 
                                                        X_train, 
                                                        y_train=None, 
                                                        **fit_params)
        else:
            raise ValueError(generic_error)
        
    else:
        # X_train, y_train are iterabls of bags NOT single-instance examples
        if isinstance(scorer, BagScorer):
            # Depreciated
            estimator = scorer.estimator_fit(estimator, 
                                             X_train,
                                             y_train, 
                                             **fit_params)
        elif isinstance(scorer, dict):
            # It is possible to pass a dictionary of scorers
            for metric, bagScorer in scorer.items():
                if isinstance(bagScorer, BagScorer):
                    estimator = bagScorer.estimator_fit(estimator, 
                                                        X_train, 
                                                        y_train, 
                                                        **fit_params)
        else:
            raise ValueError(generic_error)
        
    return estimator


@_deprecate_positional_args
def cross_validate_bag(estimator, X, y=None, *, groups=None, scoring=None, cv=None,
                   n_jobs=None, verbose=0, fit_params=None,
                   pre_dispatch='2*n_jobs', return_train_score=False,
                   return_estimator=False, error_score=np.nan):
    """Evaluate metric(s) by cross-validation and also record fit/score times.
    Read more in the :ref:`User Guide <multimetric_cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    scoring : str, callable, list/tuple, or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        See :ref:`multimetric_grid_search` for an example.
        If None, the estimator's score method is used.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
        .. versionadded:: 0.19
        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
        .. versionadded:: 0.20
    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        .. versionadded:: 0.20
    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:
            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.
    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    Single metric evaluation using ``cross_validate``
    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.33150734, 0.08022311, 0.03531764])
    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)
    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    >>> print(scores['train_r2'])
    [0.28010158 0.39088426 0.22784852]
    See Also
    ---------
    :func:`sklearn.model_selection.cross_val_score`:
        Run cross-validation for single metric evaluation.
    :func:`sklearn.model_selection.cross_val_predict`:
        Get predictions from each split of cross-validation for diagnostic
        purposes.
    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator,
            error_score=error_score)
        for train, test in cv.split(X, y, groups))

    # results is a list of dictonaries
    results = _aggregate_score_dicts(results)
    ret = {}
    ret['fit_time'] = results['fit_time']
    ret['score_time'] = results['score_time']

    # results['test_scores'] is a list of dictionaries
    test_scores = _aggregate_score_dicts(results['test_scores'])
    if return_train_score:
        train_scores = _aggregate_score_dicts(results['train_scores'])

    if return_estimator:
        fitted_estimators = results['estimator']
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = test_scores[name]
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = train_scores[name]

    return ret



def bags_2_si_generator(bags, bag_labels, sparse=False):
    """Convert a n x (m x p) array of bag instances into a k x p array of
    instances. n is the number of bags, and m is the number of instances within
    each bag. m can vary per bag. k is the total number of instances within
    all bags. k = sum (m for bag in n). p is the feature space of each instance
    inputs
    -------
    bags : (iterable) containing bags of shape (m x p) sparse arrays
    bag_labels : (iterable) containing labels assocaited with each bag. Labels
        are expanded and each instance within a bag inherits the label of the
        bag
    sparse : (bool) if True, the output instances are left as a sparse array.
        Some sklearn estimators can handle sparse feature inputs
    output
    -------
    instances, labels : (generator) """

    for bag, label in zip(bags, bag_labels):
        # Unpack bag into instances

        if sparse and isinstance(bag, csr_matrix):
            # Keep instances as sparse array
            instances = bag # Sparse array
        elif sparse and not isinstance(bag, csr_matrix):
            # Convert instances to csr matrix if sparse=True and the bag
            # is not currently a csr matrix
            instances = csr_matrix(bag) # Dense array
        elif not sparse and isinstance(bag, csr_matrix):
            # Convert csr_array to dense array
            instances = bag.toarray()
        elif not sparse and not isinstance(bag, csr_matrix):
            # Keep as dense array
            instances = bag # Dense array
        else:
            msg=('bags must be sparse array (scipy.sparse.csr_matrix) or' +
            ' dense array (np.array). Got type {}')
            raise TypeError(msg.format(type(bag)))

        labels = np.array([label].__mul__(instances.shape[0]))

        yield instances, labels


def bags_2_si( bags, bag_labels, sparse=False):
    """Convert a n x (m x p) array of bag instances into a k x p array of
    instances. n is the number of bags, and m is the number of instances within
    each bag. m can vary per bag. k is the total number of instances within
    all bags. k = sum (m for bag in n). p is the feature space of each instance
    inputs
    -------
    bags : (iterable) containing bags of shape (m x p) sparse arrays
    bag_labels : (iterable) containing labels assocaited with each bag. Labels
        are expanded and each instance within a bag inherits the label of the
        bag
    sparse : (bool) if True, the output instances are left as a sparse array.
        Some sklearn estimators can handle sparse feature inputs
    output
    -------
    instances, labels : (np.array) or (scipy.sparse.csr.csr_matrix)
    depending on 'sparse'"""

    # Initialize generator over bags
    bag_iterator = bags_2_si_generator(bags,
                                       bag_labels,
                                       sparse=sparse)

    # Initialize datasets
    instances, labels = [], []

    # Gather datasets
    for part_instances, part_labels in bag_iterator:

        instances.append(part_instances)
        labels.append(part_labels)

    # Flatten into otuput shape - [k x p] instances and [k] labels
    if sparse:
        # Row-stack sparse arrays into a sinlge  k x p sparse array
        instances = vstack(instances)
        labels = np.concatenate(labels)
    else:
        # Row-concatenate dense arrays into a single k x p array
        instances = np.concatenate(instances)
        labels = np.concatenate(labels)

    return instances, labels



class BagScorer:
    """This is a custom scoring object for use with sklearn cross validation
    model evaluation. This includes cross_validate, cross_val_score, and
    GridSearchCV

    This scoring object is specifically used for scoring bag labels predicted
    from single-instance predictions within the bag

    According to sklearn, this scorer must be
    1. It can be called with parameters (estimator, X, y), where estimator is
    the model that should be evaluated, X is validation data, and y is
    the ground truth target for X
    2. Return a floating point number that quantifies the estimator prediction
    quality on X with reference to y. If the metric is a loss then the value
    should be negated (more positive is better)

    """
    def __init__(self, scorer, sparse=False):
        """This class should NOT be passed directly as the 'scorer' argument
        to cross_validate, cross_val_score, or GridSearchCV without
        initializing the class
        
        When scorer is a dictionary of values, raise an error
        When the scorer is a callable with method _score_func then return
            the score from this scoring metric
        
        inputs
        ------
        scorer : (#TODO)
        sparse : (bool)"""

        # Initialization parameters
        self.sparse = sparse

        # Metric for scoring bags
        if isinstance(scorer, dict):
            msg='scorer cannot be type dict. Passed type {}'
            raise ValueError(msg.format(type(scorer)))

        if not hasattr(scorer, '_score_func'):
            msg=('scorer object must have callable "_score_func" with a'+
            ' signature f(y_true, y_pred)')
            raise ValueError(msg)

        self.scorer = scorer

        return None


    def __call__(self, estimator, *positional_args, **kwargs): # TODO This should accept *args, **kwargs
        """
        inputs
        -------
        estimator : () the model that should be evaluated
        X : (iterable) is validation data. It has to be an iterable of bags,
        for example an [n x (m x p)] array of bag instances. n is the number
        of bags, and m is the number of instances within each bag.
        p is the feature space of each instance
        y : () is the ground truth target for X
        outputs
        -------
        score : (float) result of score of estimator on bags
        """
        # Initialize positional and keyword arguments
        X = positional_args[0] # Validation data (for predictions)
        y = positional_args[1] # Ground truth (for metrics)

        # Test if estimator is fitted already or not
        check_is_fitted(estimator)

        # Predict on bags
        bag_predictions = self.predict_bags(estimator, X)

        # Calculate metrics - API call to scorer function
        if hasattr(self.scorer, '_score_func'):
            kwargs = self.scorer._kwargs
            ret = self.scorer._score_func(y, bag_predictions, **kwargs)

        else:
            msg=('scorer object must have callable "_score_func" with a'+
            ' signature f(y_true, y_pred)')
            raise ValueError(msg)

        return ret
    

    def estimator_fit(self, estimator, X_train, y_train=None, **fit_params):
        """The sklearn _fit_and_score method requires estimator fitting to be
        done as a separate task from scoring
        inputs
        -------
        estimator: ()
        X_train: (iterable) is validation data. It has to be an iterable of bags,
        for example an [n x (m x p)] array of bag instances. n is the number
        of bags, and m is the number of instances within each bag.
        y_train: () is the ground truth target for X
        **fit_params: ()
        outputs
        ------
        estimator : fitted estimator"""

        if y_train is None:
            msg=('Single Instance labels cannot be constructed from bags if' +
            ' y_train is None. Pass bag labels to validation')
            raise ValueError(msg)

        # Find SI data
        SI_examples, SI_labels = bags_2_si(X_train, y_train, self.sparse)

        if y_train is None:
            # This should not happen - see ValueError
            estimator.fit(SI_examples, **fit_params)
        else:
            estimator.fit(SI_examples, SI_labels, **fit_params)

        return estimator # Fitted estimator

    @staticmethod
    def reduce_bag_label(predictions, method: str = 'mode') -> Union[str, int]:
        """Determine the bag label from the single-instance classifications of 
        its members. 'mode' returns the most frequently occuring bag label
        inputs
        -------
        predictions : (iterable) of labels
        method : (str) 'mode' is only supported.
            TODO support based on label weights
        outputs
        -------
        label : (str / int) of most common prediction"""

        if method == 'mode':
            label, count = Counter(predictions).most_common(1)[0]
        else:
            method_error = "only 'mode' is supported as a reduction method." \
                "Got {}".format(method)
            raise ValueError(method_error)

        return label

    @classmethod
    def predict_bags(cls, estimator, bags, method: str ='mode'):
        """
        inputs
        ------
        estimator: ()
        bags: ()
        method: (string)
        outputs
        -------"""

        bag_predictions = []

        for bag in bags:
            # Predict labels in bag
            si_predictions = estimator.predict(bag)
            bag_prediction = cls.reduce_bag_label(si_predictions, method=method)
            bag_predictions.append(bag_prediction)

        return bag_predictions
