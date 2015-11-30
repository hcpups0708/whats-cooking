from pandas import Series, DataFrame
import pandas as pd
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


# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("train.json")
testdf = pd.read_json("test.json")
a=traindf['ingredients']
b=testdf['ingredients']
c=np.append(a,b)
d=pd.Series(c)
#traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
corpustr = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in d]       
traindf['ingredients_string'] =[' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]
#testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

#print(traindf['ingredients_string'])

#corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .67 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
vectorizertr.fit(corpustr)
tfidftr=vectorizertr.transform(traindf['ingredients_string']).todense()
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
parameters = {'weights':[[5,4,1,1,3],[5,4,1,1,4]]}
#parameters = {'n_estimators':[500],'learning_rate': [0.08],'subsample':[0.65],'objective':['multi:softmax'],'base_score':[0.05]}
#parameters = {'base_estimator__class_weight':['balanced'],'base_estimator__loss':['modified_huber'],'base_estimator__penalty':['elasticnet']}
#parameters = {'n_iter':[5,50,100]}
lsvc = LinearSVC()
svc=SVC(verbose=1,kernel='linear',probability=True)
sgd=SGDClassifier(loss='modified_huber',penalty='elasticnet')
lr = LogisticRegression(class_weight='balanced',C=10)
ovr=OneVsRestClassifier(svc,n_jobs=1)
rf=RandomForestClassifier(verbose=1,n_jobs=20,min_samples_leaf=1,n_estimators=500,oob_score=1,max_features='log2')
knn=KNeighborsClassifier(n_neighbors=1, algorithm = 'brute',weights='distance')
ab=AdaBoostClassifier(n_estimators=50)
bag=BaggingClassifier(verbose=10,base_estimator=sgd,n_estimators=50)
etc=ExtraTreesClassifier(verbose=1,n_jobs=20,n_estimators=1000)
mnb=MultinomialNB(alpha=0.025)
xgb=XGBClassifier(n_estimators=750,learning_rate=0.08,subsample=0.65,objective='multi:softmax',base_score=0.05)
vc=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf),('xgb',xgb)], voting='soft')
classifier = grid_search.GridSearchCV(vc, parameters,verbose=2)

classifier.fit(X,Y)
#classifier.fit(X_train,Y_train)

print("Start predicting")
predictions=classifier.predict(X_unknown)
print(str(len(predictions))+" results")
print(predictions)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("submission4_xgb.csv",index = False)

print("Scoring result")
for dict in classifier.grid_scores_:
    print(dict)

print("Train set accuracy: "+str(classifier.score(X_train,Y_train)))
print("Test set accuracy: "+str(classifier.score(X_test,Y_test)))