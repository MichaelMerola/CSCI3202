import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

# CLASSIFIERS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

normalset = { "Logistic Regression": LogisticRegression(),
                "Naive Bayes": GaussianNB(),
                "SVM (SVC)": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
               }

hyperset = { "SVM (SVC)": SVC(C=1000, gamma=0.0001),
             "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=96),
            }            

Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
    df = pd.read_csv(file)
    return df

def trainOnAllData(clf):
    #Use this function for part 4, once you have selected the best model
    df = readData("credit_train.csv")

    X = df.iloc[:,0:19] #data
    y = df.iloc[:,-1] #labels

    clf.fit(X, y)

    saveBestModel(clf)

############################################ CODE ###############################################

def classifierTest(classifiers, filename):
    logfile = open(filename, 'w')
    logfile.write("MODEL\t AVERAGE\t STANDARD DEV \n")

    df = readData("credit_train.csv")

    X = df.iloc[:,0:19] #data
    y = df.iloc[:,-1] #labels

    print(X.head())
    print(y.head())

    # k-fold validation #
    for name, ML in zip(classifiers.keys(), classifiers.values()):

        k = 10
        kf = KFold(n_splits=k) #fold function
        folds = kf.split(X)

        CVscores = cross_val_score(ML, X, y, scoring='roc_auc', cv=k)
        CVpredicts = cross_val_predict(ML, X, y, cv=k)

        CV_avg = np.mean(CVscores)
        CV_sd = np.std(CVscores)

        send = str(name) + "\t" + str(CV_avg) + "\t" + str(CV_sd) + "\t" + '\n'
        logfile.write(send)

        print(CV_avg, CV_sd)

    logfile.close()

def hyperTuning(model, pgrid):
    df = readData("credit_train.csv")

    X = df.iloc[:,0:19] #data
    y = df.iloc[:,-1] #labels

    trainData, testData, trainLabels, testLabels = train_test_split(X, y, test_size=0.25, random_state=42)

    print("searching random CV...")
    randomML = RandomizedSearchCV(estimator = model, param_distributions = pgrid, n_iter = 100, cv = 10)

    print("fitting model...")
    randomML.fit(trainData, trainLabels)
    acc = randomML.score(testData, testLabels)

    return acc, randomML.best_params_

def bestModelAnalysis(model, filename):
    logfile = open(filename, 'w')

    df = readData("credit_train.csv")

    X = df.iloc[:,0:19] #data
    y = df.iloc[:,-1] #labels

    trainData, testData, trainLabels, testLabels = train_test_split(X, y, test_size=0.25, random_state=42)

    # fit and test model
    model.fit(trainData, trainLabels)
    y_pred = model.predict(testData)

    acc = model.score(testData, testLabels)

    recall = metrics.recall_score(testLabels, y_pred, average="macro") 
    precision = metrics.precision_score(testLabels, y_pred, average="macro")

    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=3)
    auroc = np.mean(scores)

    matr = metrics.confusion_matrix(testLabels, y_pred).ravel()

    send = str(acc) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(auroc) + "\t" + str(matr) + '\n'
    logfile.write(send)

    logfile.close()

    # test full model
    total_pred = model.predict(X)

    # save DF file
    df['Prediction'] = total_pred
    export_csv = df.to_csv(r'bestModel.output', index = None, header=True)
    print(df.head())

############################################ TESTING ###############################################
'''
# 10-Fold Validation Classifier Test #
classifierTest(normalset, 'logFile.txt')


# SVC Tuning #
pgrid = {'C': [1, 10, 100, 1000], 
            'gamma': [0.001, 0.0001]}

svc_acc, svc_params = hyperTuning(SVC(), pgrid)
print ("score: ", svc_acc)
print(svc_params)

# RANDOM FOREST Tuning #
#number of trees
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 25)]

#max levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 12)]
max_depth.append(None)

RFgrid = {'n_estimators': n_estimators,
            'max_depth': max_depth
            }

rf_acc, rf_params = hyperTuning(RandomForestClassifier(), RFgrid)
print ("score: ", rf_acc)
print(rf_params)

# Tuning Classifier Test #
classifierTest(hyperset, 'tuneFile.txt')

# Best Model Analysis #
best = RandomForestClassifier(n_estimators=250, max_depth=96)
bestModelAnalysis(best, 'bestlog.txt')

trainOnAllData(best)
'''
