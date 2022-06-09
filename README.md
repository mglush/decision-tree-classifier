# Decision Tree Classifier Implementation in Python.
#### By Michael Glushchenko for UCSB CS165B Spring 2022 (Machine Learning).

## Table of Contents
* [Purpose](https://github.com/mglush/three-class-classifier/blob/main/README.md#purpose)
* [How To Run](https://github.com/mglush/three-class-classifier/blob/main/README.md#how-to-run)
* [Parameters and Format](https://github.com/mglush/three-class-classifier/blob/main/README.md#parameters-and-format)

## Purpose
[This repository](https://github.com/mglush/three-class-classifier) aims to build a decision tree classifier. Main purpose is to improve data-processing & data formatting skills while visualizing the decision-making process that happens in a decision-tree-classifier.

## How to Run
~~~
git clone https://github.com/mglush/decision-tree-classifier    # clone repository.
cd decision-tree-classifier                                     # enter repo folder.
python hw3.py training_filename testing_filename                # run file on a training/testing file input pair.
~~~

## Parameters and Format
**Input**:\
Two text files, one containing the training data, one containing the testing data. Check data/training*.txt for format examples.

**Output**:\
Two dictionaries, both containing the number of true positives, true negatives, false positives, false negatives, and the error rate of the classifier for the given test data. First dictionary uses results from the gini criterion being used in the decision tree, second dictionary uses results from the entropy criterion being used in the decision tree.\
Example:\
"gini":{\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"tpr": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"fpr": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"error_rate": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"accuracy": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"precision": _____,\
}\
"entropy":{\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"tpr": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"fpr": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"error_rate": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"accuracy": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"precision": _____,\
}
