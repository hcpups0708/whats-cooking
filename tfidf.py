from pandas import Series, DataFrame
import pandas as pd
import numpy as np
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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest


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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, Y).transform(X)


#clf = LinearSVC()
lr = LogisticRegression(class_weight='balanced',C=10)
ovr=OneVsRestClassifier(LogisticRegression(),n_jobs=1)
rf=RandomForestClassifier(verbose=1,n_jobs=20,min_samples_leaf=1,n_estimators=300,oob_score=1)
knn=KNeighborsClassifier(n_neighbors=1, algorithm = 'brute',weights='distance')
vc=VotingClassifier(estimators=[('lr', lr),('ovr', ovr), ('knn', knn), ('rf', rf)], voting='soft',weights=[5,2,1,3])

pipeline = Pipeline([("features", combined_features), ("vc", vc)])

parameters = {'features__pca__n_components':[500, 1000, 2000],'features__univ_select__k':[500, 1000]}

classifier = grid_search.GridSearchCV(pipeline, parameters,verbose=2)

classifier.fit(X,Y)
#classifier.fit(X_train,Y_train)

print("Start predicting")
predictions=classifier.predict(X_unknown)
print(str(len(predictions))+" results")
print(predictions)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("submission.csv",index = False)

print("Scoring result")
for dict in classifier.grid_scores_:
    print(dict)

print("Train set accuracy: "+str(classifier.score(X_train,Y_train)))
print("Test set accuracy: "+str(classifier.score(X_test,Y_test)))