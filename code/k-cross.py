import numpy as np
import pandas as pd
import random
import json
import sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn import tree
print('Libraries imported.\n')
##L = 18000 H = 12000


######## separated to have a better organizzation of results
if len(sys.argv[1:]) < 1:
        print("Error: Please insert the type of classfication (compiler / optimization) after the program name \n")
        sys.exit(1)

if (sys.argv[1] != "compiler") and (sys.argv[1] != "optimization"):
        print("Error: Please insert the type of classfication (compiler / optimization) after the program name \n")
        sys.exit(1)



"""Read data from file"""

json_dataset = []
with open('../dataset/train_dataset.jsonl') as f:
    for line in f:
        program = json.loads(line)

        """Preprocessing  text"""
        #iterate in the instruction list of the program to replace with the mnemonics only, no operands
        for i in range(len(program["instructions"])):
            instruction = program["instructions"][i].split(' ')
            program["instructions"][i] = instruction[0]     #takes the mnemonic only of the single instruction

        #convert the list of instructions to a string of instructions, to use as text
        program["instructions"] = ' '.join(program["instructions"])

        #to use a set instead of a list
        #if program not in json_dataset:

        #each program is an entry of the list
        json_dataset.append(program)

data_len = len(json_dataset)
print('File loaded: %d samples. \n' %(data_len))



"""Show a random json sample"""

id = random.randrange(0, data_len)
program_ex = json_dataset[id]
print("Random json example: #", id)
for x, y in program_ex.items():
  print(x, ": ", y)
print()



"""Use pandas to convert to a table and show 5 examples"""

dataset = pd.DataFrame(json_dataset)
print("Dataset head:")
print(dataset.head(), '\n')




if (sys.argv[1] == "optimization"):
    print("#################################### OPTIMIZATION ################################")
    """Features engineering"""

    #Each term found by the analyzer during the fit is assigned a unique integer index corresponding to a column in the resulting matrix
    vectorizer = CountVectorizer()
    print('CountVectorizer selected')

    #create features X and response y
    X = vectorizer.fit_transform(dataset.instructions)  #fit transform to apply the vectorizer changes
    y = dataset.opt


    print('X shape: ', X.shape)         #number of rows (examples attributes) and column (features)
    print('y shape: ', y.shape)         #number of rows (examples target values)
    print()



    """Split data in training set and test set"""

    #Stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3333, stratify = y )

    print("Train size: %d, Test size: %d" %(X_train.shape[0],X_test.shape[0]))
    print("Class distribution:")
    print(y_test.value_counts(), '\n')



    """Create and Fit the Model"""

    model = MultinomialNB()
    print('MultinomialNB Model created')


    """k-cross validation"""

    kfold = StratifiedKFold(n_splits=10, random_state=42)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
    print('Accuracies from each of the 10 folds using kfold:',results)
    print("Variance of kfold accuracies:",results.var())


    """Evaluation"""

    #predict will return a numpy array with the predections for all tested models
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))



    """Create Model"""

    model = tree.DecisionTreeClassifier()
    print('DecisionTreeClassifier Model created')


    """k-cross validation"""

    #already created kfold = StratifiedKFold(n_splits=10, random_state=42)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
    print('Accuracies from each of the 10 folds using kfold:',results)
    print("Variance of kfold accuracies:",results.var())


    """Evaluation"""

    #predict will return a numpy array with the predections for all tested models
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))





if (sys.argv[1] == "compiler"):
    print("#################################### COMPILER ################################")
    """Features engineering"""

    #Each term found by the analyzer during the fit is assigned a unique integer index corresponding to a column in the resulting matrix
    vectorizer = CountVectorizer()
    print('CountVectorizer selected')

    #create features X and response y
    X = vectorizer.fit_transform(dataset.instructions)  #fit transform to apply the vectorizer changes
    y = dataset.compiler

    print('X shape: ', X.shape)         #number of rows (examples attributes) and column (features)
    print('y shape: ', y.shape)         #number of rows (examples target values)
    print()



    """Split data in training set and test set"""

    #Stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3333, stratify = y )

    print("Train size: %d, Test size: %d" %(X_train.shape[0],X_test.shape[0]))
    print("Class distribution:")
    print(y_test.value_counts(), '\n')



    """Create the Model"""

    model = MultinomialNB()
    print('MultinomialN Model created')


    """k-cross validation of training set"""

    kfold = KFold(n_splits=10, random_state=42)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
    print('Accuracies from each of the 10 folds using kfold:',results)
    print("Variance of kfold accuracies:",results.var())


    """Evaluation"""

    #predict will return a numpy array with the predections for all tested models
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))



    """Create the Model"""

    model = tree.DecisionTreeClassifier()
    print('DecisionTreeClassifier Model created')


    """k-cross validation"""

    #already created kfold = StratifiedKFold(n_splits=10, random_state=42)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
    print('Accuracies from each of the 10 folds using kfold:',results)
    print("Variance of kfold accuracies:",results.var())


    """Evaluation"""

    #predict will return a numpy array with the predections for all tested models
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
