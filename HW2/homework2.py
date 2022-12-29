# %%
import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
f = open("5year.arff", 'r')

# %%
# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# %%
answers = {} # Your answers

# %%
def accuracy(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

    accuracy = (TP+TN)/(TP+FP+TN+FN)
    
    return accuracy

# %%
def BER(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    ber = 1 - 1/2 * (TPR + TNR)

    return ber


# %%
### Question 1

# %%
mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)

# %%
correct = pred == y

ber1 = BER(pred,y)
acc1 = accuracy(pred,y)

# %%
answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate

# %%
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)

# %%
correct = pred == y

ber2 = BER(pred,y)
acc2 = accuracy(pred,y)

# %%
answers['Q2'] = [acc2, ber2]

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3

# %%
random.seed(3)
random.shuffle(dataset)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# %%
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
len(Xtrain), len(Xvalid), len(Xtest)

# %%
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)

predTrain = mod.predict(Xtrain)
predValid = mod.predict(Xvalid)
predTest = mod.predict(Xtest)

berTrain = BER(predTrain,ytrain)
berValid = BER(predValid,yvalid)
berTest = BER(predTest,ytest)


# %%
answers['Q3'] = [berTrain, berValid, berTest]

# %%
assertFloatList(answers['Q3'], 3)

# %%
### Question 4

# %%
ls = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
errorTrain = [] #this is going to have the MSEtrain values of each l
errorValid = [] #this is going to have the MSEvalid values of each l
berList = []

for l in ls:
    model = linear_model.LogisticRegression(C=l, class_weight='balanced')
    model.fit(Xtrain, ytrain) #fitting the model

    predictValid = model.predict(Xvalid)


    ber = BER(predictValid,yvalid)
    berList.append(ber)

    print(ber)


# %%
answers['Q4'] = berList

# %%
assertFloatList(answers['Q4'], 9)

# %%
### Question 5

# %%
#bestModel = None
bestVal = None
bestLamb = None

ls = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
errorTrain = [] #this is going to have the MSEtrain values of each l
errorValid = [] #this is going to have the MSEvalid values of each l

for l in ls:
    model = linear_model.LogisticRegression(C=l, class_weight='balanced')
    model.fit(Xtrain, ytrain) #fitting the model

    predictValid = model.predict(Xvalid)


    ber = BER(predictValid,yvalid)

    predictTest = model.predict(Xtest)
    ber_test = BER(predictTest,ytest)

    #print("l = " + str(l) + ", BER = " + str(ber))
    if bestVal == None or ber < bestVal:
        bestVal = ber
        #bestModel = model
        bestLamb = l

bestC = bestLamb
ber5 = bestVal

print(bestC)
print(ber5)

# %%
answers['Q5'] = [bestC, ber5]

# %%
assertFloatList(answers['Q5'], 2)

# %%
### Question 6

# %%
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))

# %%
dataTrain = dataset[:9000]
dataTest = dataset[9000:]

# %%
print(dataTrain[0])

# %%
# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list) # Maps a user to the reviews they made
reviewsPerItem = defaultdict(list) # Maps an item to its reviews
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user,item = d['user_id'], d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user,item)] = d['rating']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

# %%
def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for j in usersPerItem:
        if j == i: continue
        sim = Jaccard(users, usersPerItem[j])
        #sim = Pearson(i, j) # Could use alternate similarity metrics straightforwardly
        similarities.append((sim,j))
    similarities.sort(reverse=True)
    return similarities[:N]

# %%


# %%
answers['Q6'] = mostSimilar('2767052', 10)

# %%
assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)

# %%
### Question 7

# %%
#for d in dataset:
#    user,item = d['user_id'], d['book_id']
#    reviewsPerUser[user].append(d)
#    reviewsPerItem[item].append(d)

ratingMean = sum([d['rating'] for d in dataTrain]) / len(dataTrain)

itemAverages = {}
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)


def predictRating(user,item):
    ratings = []
    simscores = []
    for d in reviewsPerUser[user]:
        #print(d)
        j = d['book_id']
        if j == item: continue
        ratings.append(d['rating'] - itemAverages[j])
        simscores.append(Jaccard(usersPerItem[item],usersPerItem[j]))
    if (sum(simscores) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,simscores)]
        return itemAverages[item] + sum(weightedRatings) / sum(simscores)
    else:
        return ratingMean


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

#alwaysPredictMean = [ratingMean for d in dataTest]
simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]

labels = [d['rating'] for d in dataTest]


mse7 = MSE(simPredictions,labels)

# %%
answers['Q7'] = mse7

# %%
assertFloat(answers['Q7'])

# %%
### Question 8

# %%

ratingMean = sum([d['rating'] for d in dataTrain]) / len(dataTrain)

def predictRating(user,item):
    ratings = []
    simscores = []
    for d in reviewsPerUser[user]:
        #print(d)
        j = d['book_id']
        if j == item: continue
        ratings.append(d['rating'] - itemAverages[j])
        simscores.append(Jaccard(itemsPerUser[user],itemsPerUser[j]))
    if (sum(simscores) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,simscores)]
        return itemAverages[user] + sum(weightedRatings) / sum(simscores)
    else:
        return ratingMean


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

#alwaysPredictMean = [ratingMean for d in dataTest]
simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]

labels = [d['rating'] for d in dataTest]

mse8 = MSE(simPredictions,labels)

# %%
answers['Q8'] = mse8

# %%
assertFloat(answers['Q8'])

# %%
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



