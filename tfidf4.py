from pandas import Series, DataFrame
import pandas as pd
import io
import json
import numpy as np
import scipy
import xgboost as xgb
import nltk
import re
import random
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
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
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, SelectKBest

country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
Xtrain_str=[]
Xtest_str=[]
X_v_t_str=[]
X_v_t=[]
Y_v_t=[]
X_unknown=[]
testDic={}
# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%



with io.open('train.json', encoding = 'utf8') as data_file:
    data = json.load(data_file)

with io.open('test.json', encoding = 'utf8') as data_file:
    test = json.load(data_file)

for dish in data:
    ing_str=""
    for ing in dish['ingredients']:
        ing_str+=" "+ing
    ing_str=WordNetLemmatizer().lemmatize(ing_str.strip())
    Xtrain_str.append(ing_str)
    Y.append(dish['cuisine'])

for dish in test:
    ing_str=""
    for ing in dish['ingredients']:
        ing_str+=" "+ing
    Xtest_str.append(WordNetLemmatizer().lemmatize(ing_str.strip()))
    testDic[dish['id']]=ing_str

#traindf['ingredients_string'] =[' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]
#testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
#testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]

#print(traindf['ingredients_string'])

#corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .67 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
vectorizertr.fit(Xtrain_str+Xtest_str)
tfidftr=vectorizertr.transform(Xtrain_str)
tfidfts=vectorizertr.transform(Xtest_str)

X = tfidftr

print(X.shape)

X_unknown = tfidfts

Xbalanced=[]
Ybalanced=[]
Xunb=[[] for i in range(20)]
labels=['irish','mexican','chinese','filipino','vietnamese','moroccan','spanish','japanese','french','greek','indian','jamaican','british','brazilian','russian','cajun_creole','korean','southern_us','thai','italian']
for idx,dish in enumerate(X.toarray()):
    for idx2,lab in enumerate(labels):
        if Y[idx] == lab:
			Xunb[idx2].append(dish)
for nameidx,samples in enumerate(Xunb):
	n_samples=3000
	print(labels[nameidx],len(samples))
	if len(samples)<=n_samples:
		for i in range(n_samples):
			Xbalanced.append(samples[i%len(samples)])
	else:
		Xbalanced=Xbalanced+random.sample(samples,n_samples)
	Ybalanced=Ybalanced+[labels[nameidx] for i in range(n_samples)]
X=scipy.sparse.csr_matrix(Xbalanced)
Y=Ybalanced
print(X.shape)
print(len(Ybalanced))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

parameters = {}
#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
#parameters = {'weights':[[5,4,1,1]]}
#parameters = {'n_estimators':[500],'learning_rate': [0.08],'subsample':[0.65],'objective':['multi:softmax'],'base_score':[0.05]}
#parameters = {'base_estimator__class_weight':['balanced'],'base_estimator__loss':['modified_huber'],'base_estimator__penalty':['elasticnet']}
#parameters = {'n_iter':[5,50,100]}

lsvc = LinearSVC()
svc=SVC(verbose=1,kernel='linear',probability=True)
sgd=SGDClassifier(loss='modified_huber',penalty='elasticnet')
lr = LogisticRegression(class_weight='balanced',C=10)
ovr=OneVsRestClassifier(svc,n_jobs=1)
rf=RandomForestClassifier(verbose=1,n_jobs=-1,min_samples_leaf=1,n_estimators=500,oob_score=1,max_features='log2')
knn=KNeighborsClassifier(n_neighbors=1, algorithm = 'brute',weights='distance')
ab=AdaBoostClassifier(n_estimators=50)
bag=BaggingClassifier(verbose=10,base_estimator=sgd,n_estimators=50)
etc=ExtraTreesClassifier(verbose=1,n_jobs=-1,n_estimators=300)
mnb=MultinomialNB(alpha=0.025)
xgb=XGBClassifier(silent=False,n_estimators=750,learning_rate=0.08,subsample=0.65,objective='multi:softmax',base_score=0.05,max_depth=15)
vc=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf)], voting='soft')
classifier = grid_search.GridSearchCV(knn, parameters,verbose=2)

classifier.fit(X,Y)
#classifier.fit(X_train,Y_train)

print("Start predicting")

predictions=classifier.predict(X_unknown)

print(str(len(predictions))+" results")
print(predictions)
out=open("ans.csv",'w')
out.write("id,cuisine\n")
i=0
for dish in test:
    out.write(str(dish['id'])+","+predictions[i]+"\n")
    i+=1
out.close()

print("Scoring result")
for dict in classifier.grid_scores_:
    print(dict)

print("Train set accuracy: "+str(classifier.score(X_train,Y_train)))
print("Test set accuracy: "+str(classifier.score(X_test,Y_test)))