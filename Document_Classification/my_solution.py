# Enter your code here. Read input from STDIN. Print output to STDOUT
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
import sys

# read in the training data (documents)
inp = [line.rstrip('\n') for line in open('trainingdata.txt')]

# cut first line: unnecessary
inp = inp[1:]

# get the category and the document separated
X = []
Y = []
for i,l in enumerate(inp):
    inp[i] = l.split(' ',1)
    X.append(inp[i][1])
    Y.append(int(inp[i][0]))

X = pd.DataFrame(X)
X.columns = ['Documents']

# some cleaning of training data beforehand
#X.Documents = X.Documents.replace('\'','')
#X.Documents = X.Documents.replace('\.','')


# get the test data
test = []
for s in sys.stdin:
    test.append(s.rstrip())
#print(test)

# cut first line: unnecessary
test = test[1:]
test = pd.DataFrame(test)
test.columns = ['Documents']

# some cleaning of test data beforehand
#test.Documents = test.Documents.replace('\'','')
#test.Documents = test.Documents.replace('\.','')

# Define a TF-IDF Vectorizer Object. 
# Remove all english stop words such as 'the', 'a'
# Limit it to 7000 features (vocabulary)
vectorizer = TfidfVectorizer(stop_words='english', max_features=8500,strip_accents='unicode',lowercase=False)
#vectorizer = CountVectorizer(stop_words='english', max_features = 8500)
#vectorizer = CountVectorizer()

# Replace NaN with an empty string
X['Documents'] = X['Documents'].fillna('')

# Construct the required TF-IDF matrix by fitting the transformation model
# and then applying the transformation
vectorizer.fit(X['Documents'])
vect_matrix = vectorizer.transform(X['Documents'])
vect_matrix_test = vectorizer.transform(test['Documents'])

# Output the shape of tfidf_matrix
#print(vect_matrix_test.shape)

# fit the classifier and predict the training and test data
# using complement NB to deal with unbalancedness
clf = MultinomialNB()
#clf = ComplementNB()
#clf = GaussianNB()
clf.fit(vect_matrix, Y)
preds = clf.predict(vect_matrix_test)
preds_tr = clf.predict(vect_matrix)

# check the balancedness [it is very unbalanced!]
#print(pd.Series(Y).value_counts())

#acc = metrics.accuracy_score(Y, preds_tr)
#score = 100*(acc-(1-acc))
#print(acc)
#print(score)


for p in preds:
    print(p)
