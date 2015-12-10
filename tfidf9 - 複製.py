from pandas import Series, DataFrame
import pandas as pd
import io
import json
import numpy as np
import scipy
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
X_it_str=[]
X_cn_str=[]
X_su_str=[]
X_ib_str=[]
X_mb_str=[]
X_it=[]
Y_it=[]
X_cn=[]
Y_cn=[]
X_su=[]
Y_su=[]
X_ib=[]
Y_ib=[]
X_mb=[]
Y_mb=[]
X_unknown=[]
testDic={}
# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

group_it=['spanish','italian','french','greek','russian']
group_cn=['chinese','filipino','korean','japanese','vietnamese','thai']
group_su=['cuisine','cajun_creole','southern_us']
group_ib=['irish','british']
group_mb=['mexican','brazilian']


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
    testDic[dish['id']]={'str':ing_str,'id':dish['id']}

	
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .67 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
vectorizertr.fit(Xtrain_str+Xtest_str)
tfidftr=vectorizertr.transform(Xtrain_str)
tfidfts=vectorizertr.transform(Xtest_str)

X = tfidftr
X_unknown = tfidfts

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


for idx,dish in enumerate(X_train.toarray()):
    if Y_train[idx] in group_it:
        X_it.append(dish)
        Y_it.append(Y_train[idx])
    if Y_train[idx] in group_cn:
        X_cn.append(dish)
        Y_cn.append(Y_train[idx])
    if Y_train[idx] in group_su:
        X_su.append(dish)
        Y_su.append(Y_train[idx])
    if Y_train[idx] in group_ib:
        X_ib.append(dish)
        Y_ib.append(Y_train[idx])
    if Y_train[idx] in group_mb:
        X_mb.append(dish)
        Y_mb.append(Y_train[idx])

X_it=scipy.sparse.csr_matrix(X_it)
X_cn=scipy.sparse.csr_matrix(X_cn)
X_su=scipy.sparse.csr_matrix(X_su)
X_ib=scipy.sparse.csr_matrix(X_ib)
X_mb=scipy.sparse.csr_matrix(X_mb)		

print("X",X.shape)
print("X_it",X_it.shape)
print("X_cn",X_cn.shape)
print("X_su",X_su.shape)
print("X_ib",X_ib.shape)
print("X_mb",X_mb.shape)

parameters = {}
parameters_vt = {}
#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
#parameters = {'weights':[[5,4,1,1]]}
#parameters = {'n_estimators':[500],'learning_rate': [0.08],'subsample':[0.65],'objective':['multi:softmax'],'base_score':[0.05]}
#parameters = {'base_estimator__class_weight':['balanced'],'base_estimator__loss':['modified_huber'],'base_estimator__penalty':['elasticnet']}
#parameters = {'n_iter':[5,50,100]}
#parameters_vt = {'weights':[[5,4,1,1]]}

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
xgb=XGBClassifier(silent=True,n_estimators=750,learning_rate=0.08,subsample=0.8,objective='multi:softmax',base_score=0.05,max_depth=25)
#xgb_vt=XGBClassifier(silent=True,n_estimators=750,learning_rate=0.08,subsample=0.8,objective='binary:logistic',num_class=2,max_depth=25)
vc=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf), ('xgb', xgb)], voting='soft',weights=[5,4,1,1,2])
vc_it=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf)], voting='soft',weights=[5,4,1,1])
vc_cn=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf)], voting='soft',weights=[5,4,1,1])
vc_su=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf)], voting='soft',weights=[5,4,1,1])
vc_ib=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf)], voting='soft',weights=[5,4,1,1])
vc_mb=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf)], voting='soft',weights=[5,4,1,1])
gs = grid_search.GridSearchCV(vc, parameters,verbose=2)
classifier_it = vc_it
classifier_cn = vc_cn
classifier_su = vc_su
classifier_ib = vc_ib
classifier_mb = vc_mb


def subpredict(preds,className,clf,Xtest):
	subidx=[]
	subX=[]
	Xarray=Xtest.toarray()
	for idx,pred in enumerate(preds):
		if pred in className:
			subidx.append(idx)
			subX.append(Xarray[idx])
	for idx2,result in enumerate(clf.predict(scipy.sparse.csr_matrix(subX))):
		preds[subidx[idx2]]=result
	return preds
	
do_gridSearch=0
local_test=1
subpredict_it=1
subpredict_cn=1
subpredict_su=1
subpredict_ib=1
subpredict_mb=1
cm=1

if do_gridSearch:
	classifier=gs
else:
	classifier=vc
	
if local_test:
	X=X_train
	Y=Y_train
	
classifier.fit(X,Y)

print("Start predicting")
if local_test:
	X_unknown=X_test
	
predictions=classifier.predict(X_unknown)

	
if subpredict_it:
	print("subpredicting it")
	classifier_it.fit(X_it,Y_it)
	predictions=subpredict(predictions,group_it,classifier_it,X_unknown)
	
if subpredict_cn:
	print("subpredicting cn")
	classifier_cn.fit(X_cn,Y_cn)
	predictions=subpredict(predictions,group_cn,classifier_cn,X_unknown)	
	
if subpredict_su:
	print("subpredicting su")
	classifier_su.fit(X_su,Y_su)
	predictions=subpredict(predictions,group_su,classifier_su,X_unknown)	
	
if subpredict_ib:
	print("subpredicting ib")
	classifier_ib.fit(X_ib,Y_ib)
	predictions=subpredict(predictions,group_ib,classifier_ib,X_unknown)
	
if subpredict_mb:
	print("subpredicting mb")
	classifier_mb.fit(X_mb,Y_mb)
	predictions=subpredict(predictions,group_mb,classifier_mb,X_unknown)	
	

if do_gridSearch:
	for dict in classifier.grid_scores_:
		print(dict)

if local_test:
	print("Scoring result")
	print("Train set accuracy: "+str(classifier.score(X_train,Y_train)))
	print("Test set accuracy: "+str(classifier.score(X_test,Y_test)))
	if cm:
		for row in confusion_matrix(Y_test,predictions,labels=['irish','mexican','chinese','filipino','vietnamese','moroccan','spanish','japanese','french','greek','indian','jamaican','british','brazilian','russian','cajun_creole','korean','southern_us','thai','italian']):
			for n in row:
				print(str(n)+" "),
			print("")
else:
	print(str(len(predictions))+" results")
	print(predictions)
	out=open("ans9.csv",'w')
	out.write("id,cuisine\n")
	i=0
	for dish in test:
		out.write(str(dish['id'])+","+predictions[i]+"\n")
		i+=1
	out.close()

#print(confusion_matrix(Y_test,Y_pred,labels=['irish','mexican','chinese','filipino','vietnamese','moroccan','spanish','japanese','french','greek','indian','jamaican','british','brazilian','russian','cajun_creole','korean','southern_us','thai','italian']))
#print(confusion_matrix(Y_test,classifier.predict(X_test)))