from collections import defaultdict
from random import *
import random

#this function made for create default dictionary
def dicCreator() :
    return defaultdict(float)

#main class
class NaiveBayes :
    #learn function write for computing the possibility of lables and states of features according to lables
    def learn(DB):
        # countstate = []
        #
        # for item in DB[0] :
        #     countstate.append([])
        #
        # for arr in DB :
        #     i = 0
        #     for item in arr :
        #         if item not in countstate[i] :
        #             countstate[i].append(item)
        #         i = i + 1
        # return countstate

        possibilityAtt = []
        possibilityLabel = defaultdict(float)

        for i in range(0,9) :
            possibilityAtt.append(defaultdict(dicCreator))

        for instance in DB :
            i = 0
            possibilityLabel[instance[-1]] += 1
            for feature in instance[:-1] :
                possibilityAtt[i][feature][instance[-1]] += 1
                i = i + 1

        # return possibilityAtt
        # return possibilityLable
        i = 0
        for feature in possibilityAtt :
            for k1,v1 in feature.items() :
                for k2,v2 in v1.items() :
                    possibilityAtt[i][k1][k2] /= possibilityLabel[k2]
            i = i+ 1

        for k,v in possibilityLabel.items() :
            possibilityLabel[k] /= len(DB)

        possibilityAtt.append(possibilityLabel)
        return possibilityAtt

    #this function write for compute the possibility of each lable for a new instance of data
    def predict(possibility , instance):
        possibilityAtt = possibility[:-1]
        possibilityLabel = possibility[-1]
        resault = defaultdict(float)
        for k,v in possibilityLabel.items():
            i = 0
            resault[k] = v
            for feature in instance :
                resault[k] *= possibilityAtt[i][feature][k]
                i += 1
        return (max(resault),resault)


#this function write for compute that how much our guess is close to reality
def crossValidate(data):
    accuracy = []
    partSize = int(len(data) / 6)

    for i in range(0,6):
        test = data[(i * partSize) : ((i + 1) * partSize)]
        train = data[:(i * partSize)] + data[((i + 1) * partSize):]
        possibilitys = NaiveBayes.learn(train)

        counter = 0
        for instance in test:
            predict,tempPredict = NaiveBayes.predict(possibilitys, instance[:-1])
            if predict == instance[-1] :
                counter += 1
        counter /= partSize
        accuracy.append(counter)
    return (accuracy)




file = open("tictactoe.txt")
data=[]

for items in file :
    items = items.replace("\n","")
    data.append(items.split(","))
shuffle(data)

possibility =NaiveBayes.learn(data)
print("feature posibility :",possibility[:-1])
print("label posibility :" ,possibility[-1])

randomData = random.choice(data)
print("Random selected data :",randomData)
print("predict :",NaiveBayes.predict(possibility,randomData[:-1]))

accuracy = crossValidate(data)
print("Cross Validation:", accuracy)

avrage = sum(accuracy) / len(accuracy)
print("average :", avrage)
# a,b,c = crossValidate(data)
# print('a',a)
# print('b',b)
# print('c',c)
