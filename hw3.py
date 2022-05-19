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

    # 1) parse the input data into appropriate format.
    [train_features, train_targets] = parse_file(training_file)
    [test_features, test_targets] = parse_file(testing_file)

    # print("FEATURES\n", train_features)
    # print("TARGETS\n", train_targets)

    # 2) train the decision tree classifiers for gini and entropy.
    gini_dtc = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
    gini_dtc = gini_dtc.fit(train_features, train_targets)
    entropy_dtc = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    entropy_dtc = entropy_dtc.fit(train_features, train_targets)

    # print("PREDICTED GINI\n", gini_dtc.predict(test_features))
    # print("PREDICTED ENTROPY\n", entropy_dtc.predict(test_features))
    # print("ACTUAL\n", test_targets)

    # 3) run the trained decision tree on test data.
    gini_result = gini_dtc.predict(test_features)
    entropy_result = entropy_dtc.predict(test_features)

    # 4) calculate the results!
    gini_fp = len([i for i in (gini_result - test_targets) if i > 0])
    gini_fn = len([i for i in (gini_result - test_targets) if i < 0])
    gini_tp = len([i for i in range(len(test_targets)) if gini_result[i] == test_targets[i] == 1])
    gini_tn = len([i for i in range(len(test_targets)) if gini_result[i] == test_targets[i] == 0])
    gini_error_rate = (gini_fn + gini_fp) / len(test_targets)

    entropy_fp = len([i for i in (entropy_result - test_targets) if i > 0])
    entropy_fn = len([i for i in (entropy_result - test_targets) if i < 0])
    entropy_tp = len([i for i in range(len(test_targets)) if entropy_result[i] == test_targets[i] == 1])
    entropy_tn = len([i for i in range(len(test_targets)) if entropy_result[i] == test_targets[i] == 0])
    entropy_error_rate = (gini_fn + gini_fp) / len(test_targets)

    # 5) return the desired dictionaries.
    return {
            "gini":{
                'True positives': gini_tp,
                'True negatives': gini_tn,
                'False positives': gini_fp,
                'False negatives': gini_fn,
                'Error rate':gini_error_rate
                },
            "entropy":{
                'True positives': entropy_tp,
                'True negatives': entropy_tn,
                'False positives': entropy_fp,
                'False negatives': entropy_fn,
                'Error rate':entropy_error_rate
                }
            }

def parse_file(file):
    with file:
        # snag everything from the file, put it into a list of lists called data.
        # at the end, strip away the names of each feautre, they're unnecessary for this.
        data = [[y for y in x.strip().split(" ")] for x in file][1:]
        # convert all the data strings to integers.
        data = [[int(x) for x in y] for y in data]
        # remove the column that enumerates data observations,
        # and split the rest of the data into feautures and targets at the same time.
        # (the targets are located at the last column of data).
        features = [[data[j][i] for i in range(1, len(data[0]) - 1)] for j in range(len(data))]
        targets = [data[j][-1] for j in range(len(data))]
        return [np.array(features), np.array(targets)]


#######
# The following functions are provided for you to test your classifier.
#######

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()

