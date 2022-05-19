# Homework 3
# By Michael Glushchenko, 9403890.
# For UCSB CS165B, Spring 2022.

import numpy as np
from sklearn import tree

def run_train_test(training_file, testing_file):
    """
    Inputs:
        training_file: file object returned by open('training.txt', 'r')
        testing_file: file object returned by open('test1/2/3.txt', 'r')

    Output:
        Dictionary of result values for gini and entropy decision tree classifier.
        
        Example:
            return {
    			"gini":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00
    				},
    			"entropy":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00}
    				}
    """

    # parse the input data into appropriate format.
    [train_features, train_targets] = parse_file(training_file)
    [test_features, test_targets] = parse_file(testing_file)

    # create decision trees.
    gini_dtc = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
    entropy_dtc = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)

    # train the decision trees, run the trained decision trees on test data.
    gini_result = gini_dtc.fit(train_features, train_targets).predict(test_features)
    entropy_result = entropy_dtc.fit(train_features, train_targets).predict(test_features)

    # calculate the confusion matrix.
    gini_fp = len([i for i in (gini_result - test_targets) if i > 0])
    gini_fn = len([i for i in (gini_result - test_targets) if i < 0])
    gini_tp = len([i for i in range(len(test_targets)) if gini_result[i] == test_targets[i] == 1])
    gini_tn = len([i for i in range(len(test_targets)) if gini_result[i] == test_targets[i] == 0])

    entropy_fp = len([i for i in (entropy_result - test_targets) if i > 0])
    entropy_fn = len([i for i in (entropy_result - test_targets) if i < 0])
    entropy_tp = len([i for i in range(len(test_targets)) if entropy_result[i] == test_targets[i] == 1])
    entropy_tn = len([i for i in range(len(test_targets)) if entropy_result[i] == test_targets[i] == 0])

    # return the desired dictionaries.
    return {
            "gini":{
                'True positives': gini_tp,
                'True negatives': gini_tn,
                'False positives': gini_fp,
                'False negatives': gini_fn,
                'Error rate': (gini_fn + gini_fp) / len(test_targets)
                },
            "entropy":{
                'True positives': entropy_tp,
                'True negatives': entropy_tn,
                'False positives': entropy_fp,
                'False negatives': entropy_fn,
                'Error rate': (entropy_fn + entropy_fp) / len(test_targets)
                }
            }

def parse_file(file):
    """
    Input: file to be parsed.
        Example:
        # Budget Genre FamousActors Director GoodMovie
        1   0      0        1          0        0
        2   0      0        1          1        0
        3   1      0        1          0        1
    
    Output: two numpy arrays containing the feautures and targets within the file.
        Example:

        return [    [[0 0 1 0]  , [0 0 1]    ]
                     [0 0 1 1]
                     [1 0 1 0]]
    """
    with file:
        data = [[y for y in x.strip().split(" ")] for x in file][1:]
        data = [[int(x) for x in y] for y in data]
        features = [[data[j][i] for i in range(1, len(data[0]) - 1)] for j in range(len(data))]
        targets = [data[j][-1] for j in range(len(data))]
        return [np.array(features), np.array(targets)]

if __name__ == "__main__":
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()

