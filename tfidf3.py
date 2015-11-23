from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("test.json")
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

#print(traindf['ingredients_string'])

corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .67 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
#print(vectorizertr.get_feature_names())
sw=vectorizertr.get_stop_words()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)

X = tfidftr

print(X.shape)

Y = traindf['cuisine']

X_unknown = tfidfts

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
#parameters = {'weights':[[5,2,1,3]]}
parameters = {'weights':[[4,5,2,1,3]]}
svc = SVC(probability=True)
lr = LogisticRegression(class_weight='balanced',C=10)
ovr=OneVsRestClassifier(LogisticRegression(),n_jobs=1)
rf=RandomForestClassifier(verbose=1,n_jobs=20,min_samples_leaf=1,n_estimators=500,oob_score=1,max_features='log2')
knn=KNeighborsClassifier(n_neighbors=1, algorithm = 'brute',weights='distance')
ab=AdaBoostClassifier(n_estimators=50)
bag=BaggingClassifier(verbose=10,base_estimator=LogisticRegression(solver='newton-cg'),n_estimators=5,max_samples=1.0,max_features=0.1)
etc=ExtraTreesClassifier(verbose=1,n_jobs=20,n_estimators=1000)
mnb=MultinomialNB(alpha=0.025)
bnb=BernoulliNB(0.025)
vc=VotingClassifier(estimators=[('etc',etc),('lr', lr),('ovr', ovr), ('knn', knn), ('rf', rf)], voting='soft')
classifier = grid_search.GridSearchCV(vc, parameters,verbose=2)

classifier.fit(X,Y)
#classifier.fit(X_train,Y_train)

print("Start predicting")
predictions=classifier.predict(X_unknown)
print(str(len(predictions))+" results")
print(predictions)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("submission3.csv",index = False)

print("Scoring result")
for dict in classifier.grid_scores_:
    print(dict)

print("Train set accuracy: "+str(classifier.score(X_train,Y_train)))
print("Test set accuracy: "+str(classifier.score(X_test,Y_test)))