# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:37:16 2020

@author: johnvorsten
vorstenjohn@gmail.com
"""

# Python imports
import setuptools


#%%


long_description = """# Bag cross validation

Multiple instance labeling (MIL) refers to labeling data arranged in sets, or bags.  In MIL supervised learning, labels are known for bags of instances, and the goal is to assign bag-level labels to unobserved bags.

A simple solution to the MIL problem is to treat each instance in a bag as a single instance (SI) that inherits the label of its bag.  Each SI in a bag are labeled with a single-instance estimator, and the bag label is reduced from some metric (mode, threshold, presence) of the SI observations.  Official terms include *presence based, threshold based, or count based concepts* (see A two-level learning method for generalized multi-instance problems by Weidmann Nils et. al.).

scikit-learn is a popular tool for data analysis, and includes APIs for SI estimators.  It includes a convenient API for evaluating SI estimators, namely `cross_validate`

This package uses sklearn's cross_validate method and extends it to MIL for SI estimators."""

short_description="""
Cross validation for labeling bag-level data with single-instance inference
Commonly called Multiple Instance Learning (MIL)
See
Marc Carbonneau and Veronika Cheplygina and Eric Granger and Ghyslain Gagnon
Multiple Instance Learning : A Survey of Problem Characteristics and Applications
Pattern Recognition, Volume 77.1 Pg 329-353, 2018
DOI={10.1016/j.patcog.2017.10.009}
"""

setuptools.setup(
    name="bag-cross-validate-johnv", # Replace with your own username
    version="0.0.1",
    author="John Vorsten",
    author_email="vorstenjohn@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)