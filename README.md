# Bag cross validation

Multiple instance labeling (MIL) refers to labeling data arranged in sets, or bags.  In MIL supervised learning, labels are known for bags of instances, and the goal is to assign bag-level labels to unobserved bags.

A simple solution to the MIL problem is to treat each instance in a bag as a single instance (SI) that inherits the label of its bag.  Each SI in a bag are labeled with a single-instance estimator, and the bag label is reduced from some metric (mode, threshold, presence) of the SI observations.  Official terms include *presence based, threshold based, or count based concepts* (see A two-level learning method for generalized multi-instance problems by Weidmann Nils et. al.).

scikit-learn is a popular tool for data analysis, and includes APIs for SI estimators.  It includes a convenient API for evaluating SI estimators, namely [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn-model-selection-cross-validate)

This package uses sklearn's cross_validate method and extends it to MIL for SI estimators.

