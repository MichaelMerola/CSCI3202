import numpy as np
import pandas as pd
import utils
# import csv
import random

from sklearn.neural_network import MLPClassifier

# Perfect Instances
five = [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1,
        1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
two = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1,
       1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
patterns = [five, two]
labels = ['five', 'two']


def loadGeneratedData(length):
    df = pd.read_csv("mime4339-TrainingData.csv")
    X = df.iloc[:,0:length] #data
    y = df.iloc[:,-1] #labels

    return X, y

def distort_input(instance, percent_distortion):

    # percent distortion should be a float from 0-1
    # Should return a distorted version of the instance, relative to distoriton Rate
    new_instance = []

    for i in instance:
        r = random.uniform(0, 1)

        if (r <= percent_distortion):
            #flip
            if (i == 0):
                i = 1
            else:
                i = 0

        new_instance.append(i)

    return new_instance


class HopfieldNetwork:
    def __init__(self, size):
        self.h = np.zeros([size, size])

    def addSinglePattern(self, p):
        # Update the hopfield matrix using the passed pattern
    
        for row in range(len(p)):
            for col in range(len(p)):
                if (row < col):
                    val = (2*p[row]-1)*(2*p[col]-1)
                    self.h[row][col] += val  #update weight
                    self.h[col][row] += val  #update mirror
                #keep diagonal at zeros
                elif (row == col):   
                    self.h[row][col] = 0

    def fit(self, patterns):
        # for each pattern
        # Use your addSinglePattern function to learn the final h matrix
        for pat in patterns:
            self.addSinglePattern(pat)

    def retrieve(self, inputPattern):
        # Use your trained hopfield network to retrieve and return a pattern based on the
        # input pattern.
        # HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
        # has generally better convergence properties than synchronous updating.

        nodeLength = len(inputPattern)
        att = inputPattern
        change = 1

        while (change > 0): #while there are still changes in the attractor

            change = 0
            order = np.random.choice(nodeLength, nodeLength, replace=False)
            # print("ORDER: ", order)

            for i in order:
                node = self.h[i]
                # print("NODE ", i, ": ", node)
                V = np.dot(node, att)
                # print("V RESULT = ", V)
                if (V >= 0):
                    V = 1
                else:
                    V = 0
                #compare change
                if (V != att[i]): 
                    # print("**node changes**")
                    att[i] = V
                    change += 1
                # print("ATT: ", att)
            #end for
        #end while
        
        return att

    def classify(self, inputPattern):
        # Classify should consider the input and classify as either, five or two
        # You will call your retrieve function passing the input
        # Compare the returned pattern to the 'perfect' instances
        # return a string classification 'five', 'two' or 'unknown'

        out = self.retrieve(inputPattern)
        c = ''

        if (list(out) == five):
            c = 'five'
        elif (list(out) == two):
            c = 'two'
        else:
            c = 'unknown'

        print(c)
        return c


if __name__ == "__main__":
    utils.visualize(five)
    utils.visualize(two)
    length = 25
    #X, y = loadGeneratedData(length)

    #print("**HOPFIELD MODEL**")
    hopfieldNet = HopfieldNetwork(length)

    # hopfieldNet.fit(patterns)

    # for test in X.to_numpy():
    #     hopfieldNet.classify(test)

    # MLP
    # print("**MLP CLASSIFIER**")
    # model = MLPClassifier()

    # model.fit(patterns, labels)
    # y_pred = model.predict(X)
    # acc = model.score(X, y)
    # print("prediction: ", y_pred)
    # print("score: ", acc)

    #print()
    # Distortion
    #print("**DISTORTED HOPFIELD**")
    # hop_acc = {}
    # for rate in np.arange(0, 0.5, 0.01):

    #     acc = 0
    #     for i, test in enumerate(X.to_numpy()):
    #         print(test)
    #         dist = distort_input(test, rate)
    #         c = hopfieldNet.classify(dist)
    #         if (c == y[i]):
    #             acc += 1
        
    #     score = acc/len(y)
    #     hop_acc[rate] = score
    
    # w = csv.writer(open("hop-dict.csv", "w"))
    # for key, val in hop_acc.items():
    #     w.writerow([key, val])
    #print("**DISTORTED MLP**")

    # model.fit(patterns, labels)
    # y_pred = model.predict(X)
    # acc = model.score(X, y)

    # mlp_acc = {}
    # for rate in np.arange(0, 0.5, 0.01):

    #     X_dis = X.to_numpy()
    #     for i, test in enumerate(X_dis):
    #         print(test)
    #         dist = distort_input(test, rate)
    #         X_dis[i] = dist
        
    #     pred = model.predict(X_dis)
    #     mlp_acc[rate] = model.score(X, y)
    
    # w = csv.writer(open("mlp-dict.csv", "w"))
    # for key, val in mlp_acc.items():
    #     w.writerow([key, val])


# end main

# TESTING CUSTOM DATA
# utils.visualize([1,1,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0])
# utils.visualize([0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0])
# utils.visualize([0,1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0])
# utils.visualize([1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0])
# utils.visualize([0,1,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1])
# utils.visualize([0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,1,1,1])
# utils.visualize([0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0])
# utils.visualize([0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,0])
