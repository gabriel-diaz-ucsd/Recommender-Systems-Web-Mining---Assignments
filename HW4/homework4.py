# %%
import gzip
import math
import numpy
import random
import sklearn
import string
from collections import defaultdict
from gensim.models import Word2Vec
from nltk.stem.porter import *
from sklearn import linear_model
from sklearn.manifold import TSNE
import dateutil

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
dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%
answers = {}

# %%
### Question 1

# %%
print(dataset[0])

# %%
trainSet = dataset[:10000]
testSet = dataset[10000:]

trainReviewText = [d['review_text'] for d in trainSet]
#print(trainSet[0])
#print(trainReviewText[0])

# %%
uniCount = defaultdict(int)
biCount = defaultdict(int)
wordCount = defaultdict(int)
punctuation = set(string.punctuation)

for review in trainReviewText:
    r = ''.join([c for c in review.lower() if not c in punctuation])
    ws = r.split()
    ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
    for w_u in ws:
        uniCount[w_u] += 1

    for w_b in ws2:
        biCount[w_b] += 1

    for w in ws + ws2:
        wordCount[w] += 1

uCounts = [(uniCount[w], w) for w in uniCount]
uCounts.sort()
uCounts.reverse()

bCounts = [(biCount[w], w) for w in biCount]
bCounts.sort()
bCounts.reverse()

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

#print(uCounts)
#print(bCounts)

#this has unigrams and bigrams:
#print(counts)

unigrams = [x[1] for x in uCounts[:1000]]
bigrams = [x[1] for x in bCounts[:1000]]
words = [x[1] for x in counts[:1000]]

#print(words)


# %%
uniId = dict(zip(unigrams, range(len(unigrams))))
uniSet = set(unigrams)

biId = dict(zip(bigrams, range(len(bigrams))))
biSet = set(bigrams)

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

# %%

#feature for using unigrams
def feature1(datum):
    feat = [0]*len(unigrams)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    ws = r.split()
    for w in ws:
        if w in unigrams:
            feat[uniId[w]] += 1
    feat.append(1) #offset
    return feat


#feature for using bigrams
def feature2(datum):
    feat = [0]*len(bigrams)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    ws = r.split()
    ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
    for w in ws2:
        if w in bigrams:
            feat[biId[w]] += 1
    feat.append(1) #offset
    return feat


#feature for using the mix of unigrams and bigrams
def feature3(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    ws = r.split()
    ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
    for w in ws + ws2:
        if w in words:
            feat[wordId[w]] += 1
    feat.append(1) #offset
    return feat

# %%
Xtrain_a = [feature1(d) for d in trainSet]
Xtrain_b = [feature2(d) for d in trainSet]
Xtrain_c = [feature3(d) for d in trainSet]

y_train = [d['rating'] for d in trainSet]

Xtest_a = [feature1(d) for d in testSet]
Xtest_b = [feature2(d) for d in testSet]
Xtest_c = [feature3(d) for d in testSet]

y_test = [d['rating'] for d in testSet]

# %%
clf_a = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
clf_a.fit(Xtrain_a, y_train)
theta_a = clf_a.coef_
preds_a = clf_a.predict(Xtest_a)

# %%
clf_b = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
clf_b.fit(Xtrain_b, y_train)
theta_b = clf_b.coef_
preds_b = clf_b.predict(Xtest_b)

# %%
clf_c = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
clf_c.fit(Xtrain_c, y_train)
theta_c = clf_c.coef_
preds_c = clf_c.predict(Xtest_c)

# %%
uniSort = list(zip(theta_a[:-1], unigrams))
uniSort.sort()

biSort = list(zip(theta_b[:-1], bigrams))
biSort.sort()

wordSort = list(zip(theta_c[:-1], words))
wordSort.sort()

# %%
def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

# %%
for q in 'Q1a', 'Q1b', 'Q1c':
    if(q == 'Q1a'):
        mse = MSE(preds_a, y_test)
        answers[q] = [float(mse), [x[1] for x in uniSort[:5]], [x[1] for x in uniSort[-5:]]]
        print(mse)
        print(answers[q])
    elif(q == 'Q1b'):
        mse = MSE(preds_b, y_test)
        answers[q] = [float(mse), [x[1] for x in biSort[:5]], [x[1] for x in biSort[-5:]]]
        print(mse)
        print(answers[q])
    else:
        mse = MSE(preds_c, y_test)
        answers[q] = [float(mse), [x[1] for x in wordSort[:5]], [x[1] for x in wordSort[-5:]]]
        print(mse)
        print(answers[q])

# %%
for q in 'Q1a', 'Q1b', 'Q1c':
    assert len(answers[q]) == 3
    assertFloat(answers[q][0])
    assert [type(x) for x in answers[q][1]] == [str]*5
    assert [type(x) for x in answers[q][2]] == [str]*5

# %%
### Question 2

# %%
wordCount2 = defaultdict(int)
for review in trainReviewText:
    r = ''.join([c for c in review.lower() if not c in punctuation])
    for w in r.split():
        wordCount2[w] += 1

# %%
counts2 = [(wordCount2[w], w) for w in wordCount2]
counts2.sort()
counts2.reverse()

words2 = [x[1] for x in counts2[:1000]]

# %%
df = defaultdict(int)
for review in trainReviewText:
    r = ''.join([c for c in review.lower() if not c in punctuation])
    for w in set(r.split()):
        df[w] += 1

# %%
#"compared to thw first review in the dataset"
rev = trainReviewText[0]
#print(rev)

# %%
tf = defaultdict(int)
r = ''.join([c for c in rev.lower() if not c in punctuation])
for w in r.split():
    # Note = rather than +=, different versions of tf could be used instead
    tf[w] = 1
    
tfidf = dict(zip(words2,[tf[w] * math.log2(len(trainReviewText) / df[w]) for w in words2]))
tfidfQuery = [tf[w] * math.log2(len(trainReviewText) / df[w]) for w in words2]

# %%
maxTf = [(tf[w],w) for w in words2]
maxTf.sort(reverse=True)
maxTfIdf = [(tfidf[w],w) for w in words2]
maxTfIdf.sort(reverse=True)

# %%
#Cosine Similarity:
def Cosine(x1,x2):
    numer = 0
    norm1 = 0
    norm2 = 0
    for a1,a2 in zip(x1,x2):
        numer += a1*a2
        norm1 += a1**2
        norm2 += a2**2
    if norm1*norm2:
        return numer / math.sqrt(norm1*norm2)
    return 0

# %%
similarities = []
for rev2 in trainReviewText:
    tf = defaultdict(int)
    r = ''.join([c for c in rev2.lower() if not c in punctuation])
    for w in r.split():
        # Note = rather than +=
        tf[w] = 1
    tfidf2 = [tf[w] * math.log2(len(trainReviewText) / df[w]) for w in words2]
    similarities.append((Cosine(tfidfQuery, tfidf2), rev2))

# %%
similarities.sort(reverse=True)
#print(similarities[0][1])
sim = similarities[0][0]
review = similarities[0][1]

# %%
answers['Q2'] = [sim, review]

# %%
assert len(answers['Q2']) == 2
assertFloat(answers['Q2'][0])
assert type(answers['Q2'][1]) == str

# %%
### Question 3

# %%
reviewsPerUser = defaultdict(list)

# %%
for d in dataset:
    reviewsPerUser[d['user_id']].append((dateutil.parser.parse(d['date_added']), d['book_id']))

# %%
reviewLists = []
for u in reviewsPerUser:
    rl = list(reviewsPerUser[u])
    rl.sort()
    reviewLists.append([x[1] for x in rl])

# %%
model10 = Word2Vec(reviewLists,
                 min_count=1, # Words/items with fewer instances are discarded
                 vector_size=10, # Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model

# %%
sims = model10.wv.similar_by_word('18471619')
similarities = sims[:5]
print(similarities)

# %%
answers['Q3'] = similarities # probably want model10.wv.similar_by_word(...)[:5]

# %%
assert len(answers['Q3']) == 5
assert [type(x[0]) for x in answers['Q3']] == [str]*5
assertFloatList([x[1] for x in answers['Q3']], 5)

# %%
### Question 4

# %%
print(model10.wv['18471619']) #numpy vector of a book id

# %%
#print(dataset[0])
#user_id
#book_id
#review_id
#rating
#review_text
#date_added
#date_updated
#read_at
#started_at
#n_votes
#n_comments

# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list) # Maps a user to the reviews they made
reviewsPerItem = defaultdict(list) # Maps an item to its reviews
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in trainSet:
    user,item = d['user_id'], d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user,item)] = d['rating']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)

# %%
ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

itemAverages = {}
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)


def predictRating(user,item):
    ratings = []
    simscores = []
    for d in reviewsPerUser[user]:
        #print(d)
        j = d['book_id'] #you're getting the id of the current book that user read
        if j == item: continue
        ratings.append(d['rating'] - itemAverages[j])
        simscores.append(Cosine(model10.wv[item],model10.wv[j]))
    if (sum(simscores) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,simscores)]
        return itemAverages[item] + sum(weightedRatings) / sum(simscores)
    else:
        return ratingMean


simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataset[:1000]]
labels = [d['rating'] for d in dataset[:1000]]

mse4 = MSE(simPredictions,labels)

# %%
print(mse4)

# %%
answers['Q4'] = mse4

# %%
assertFloat(answers['Q4'])

# %%
### Q5

# %%
model5 = Word2Vec(reviewLists,
                 min_count=1, # Words/items with fewer instances are discarded
                 vector_size=8, # Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model

ratingMean2 = sum([d['rating'] for d in dataset]) / len(dataset)

itemAverages2 = {}
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages2[i] = sum(rs) / len(rs)


def predictRating2(user,item):
    ratings2 = []
    simscores2 = []
    for d in reviewsPerUser[user]:
        #print(d)
        j = d['book_id'] #you're getting the id of the current book that user read
        if j == item: continue
        ratings2.append(d['rating'] - itemAverages2[j])
        simscores2.append(Cosine(model5.wv[item],model5.wv[j]))
    if (sum(simscores2) > 0):
        weightedRatings2 = [(x*y) for x,y in zip(ratings2,simscores2)]
        return itemAverages[item] + sum(weightedRatings2) / sum(simscores2)
    else:
        return ratingMean


simPredictions2 = [predictRating2(d['user_id'], d['book_id']) for d in dataset[:1000]]
labels2 = [d['rating'] for d in dataset[:1000]]

mse5 = MSE(simPredictions2,labels2)

# %%
print(mse5)

# %%
answers['Q5'] = ["For my solution, I changed the vector size from 10 to 8",
                 mse5]

# %%
assert len(answers['Q5']) == 2
assert type(answers['Q5'][0]) == str
assertFloat(answers['Q5'][1])

# %%
f = open("answers_hw4.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



