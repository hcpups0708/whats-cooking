from pandas import Series, DataFrame
import pandas as pd
import io
import json
import numpy as np
import xgboost as xgb
import nltk
import re
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
from sklearn.metrics import confusion_matrix

country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
Xtrain_str=[]
Xtest_str=[]
X_v_t_str=[]
X_it_sp_str=[]
X_v_t=[]
Y_v_t=[]
X_it_sp=[]
Y_it_sp=[]
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
    if dish['cuisine']=='vietnamese' or dish['cuisine']=='thai':
        X_v_t_str.append(ing_str)
        Y_v_t.append(dish['cuisine'])
    if dish['cuisine']=='italian' or dish['cuisine']=='spanish':
        X_it_sp_str.append(ing_str)
        Y_it_sp.append(dish['cuisine'])
    Y.append(dish['cuisine'])

for dish in test:
    ing_str=""
    for ing in dish['ingredients']:
        ing_str+=" "+ing
    Xtest_str.append(WordNetLemmatizer().lemmatize(ing_str.strip()))
    testDic[dish['id']]={'str':ing_str,'id':dish['id']}

	
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .67 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
vectorizertr.fit(Xtrain_str+Xtest_str)
tfidftr=vectorizertr.transform(Xtrain_str)
tfidftr_v_t=vectorizertr.transform(X_v_t_str)
tfidftr_it_sp=vectorizertr.transform(X_it_sp_str)
tfidfts=vectorizertr.transform(Xtest_str)

X = tfidftr
X_v_t=tfidftr_v_t
X_it_sp=tfidftr_it_sp
X_unknown = tfidfts
print("X",X.shape)
print("X_v_t",X_v_t.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_v_t, Y_v_t, test_size=0.25, random_state=0)

parameters = {}
#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
#parameters = {'weights':[[5,4,1,1,2]]}
#parameters = {'n_estimators':[500],'learning_rate': [0.08],'subsample':[0.65],'objective':['multi:softmax'],'base_score':[0.05]}
#parameters = {'base_estimator__class_weight':['balanced'],'base_estimator__loss':['modified_huber'],'base_estimator__penalty':['elasticnet']}
#parameters = {'n_iter':[5,50,100]}
#parameters = {'weights':[[5,4,1,2]]}

lsvc = LinearSVC(multi_class='crammer_singer')
svc=SVC(verbose=1,kernel='linear',probability=True)
sgd=SGDClassifier(loss='modified_huber',penalty='elasticnet')
lr = LogisticRegression(class_weight='balanced',C=10)
ovr=OneVsRestClassifier(svc,n_jobs=1)
rf=RandomForestClassifier(verbose=1,n_jobs=-2,min_samples_leaf=1,n_estimators=500,oob_score=1,max_features='log2')
knn=KNeighborsClassifier(n_neighbors=1, algorithm = 'brute',weights='distance')
ab=AdaBoostClassifier(n_estimators=50)
bag=BaggingClassifier(verbose=10,base_estimator=lsvc,n_estimators=50)
etc=ExtraTreesClassifier(verbose=1,n_jobs=-2,n_estimators=1000)
mnb=MultinomialNB(alpha=0.025)
xgb=XGBClassifier(silent=True,n_estimators=750,learning_rate=0.08,subsample=0.8,objective='binary:logistic',base_score=0.05,max_depth=25)
vc=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf),('xgb',xgb)], voting='soft')
vc_v_t=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('rf', rf),('xgb',xgb)], voting='soft')
classifier = grid_search.GridSearchCV(rf, parameters,verbose=2)
#classifier_v_t = grid_search.GridSearchCV(vc_v_t, parameters_v_t,verbose=2)

#classifier.fit(X,Y)
#classifier_v_t.fit(X_train,Y_train)
classifier.fit(X_train,Y_train)

print("Start predicting")

predictions=classifier.predict(X_unknown)


#for idx,pred in enumerate(predictions):
#	if pred=='vietnamese' or pred=='thai':
#		predictions[idx]=classifier_v_t.predict(X[idx])

print(str(len(predictions))+" results")
print(predictions)
out=open("ans10.csv",'w')
out.write("id,cuisine\n")
i=0
for dish in test:
    out.write(str(dish['id'])+","+predictions[i]+"\n")
    i+=1
out.close()

print("Scoring result")
for dict in classifier.grid_scores_:
    print(dict)
#print(confusion_matrix(Y_test,classifier.predict(X_test),labels=['vietnamese','thai']))
print(confusion_matrix(Y_test,classifier.predict(X_test)))
print("Train set accuracy: "+str(classifier.score(X_train,Y_train)))
print("Test set accuracy: "+str(classifier.score(X_test,Y_test)))