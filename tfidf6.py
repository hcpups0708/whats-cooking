from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import xgboost as xgb
import nltk
import re
import io
import json
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
from xgboost import XGBClassifier
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, SelectKBest
from sklearn.decomposition import PCA
country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
X_unknown=[]
#load from json file
with io.open('train.json', encoding = 'utf8') as data_file:
	data = json.load(data_file)

with io.open('test.json', encoding = 'utf8') as data_file:
	test = json.load(data_file)

with io.open('spice.txt', encoding = 'utf8') as f:
	spices = f.read().splitlines()

with io.open('meat.txt', encoding = 'utf8') as f:
	meat = f.read().splitlines()

with io.open('seafood.txt', encoding = 'utf8') as f:
	seafood = f.read().splitlines()

with io.open('veg.txt', encoding = 'utf8') as f:
	veg = f.read().splitlines()

subdata=data                        #use full data for training
#subdata=random.sample(data,20)   #randomly select n data for training


#count and add ingredients / countries
for dish in subdata:
	if dish['cuisine'] in country:
		country[dish['cuisine']]+=1
	else:
		country[dish['cuisine']]=1
	for ing in dish['ingredients']:
		ingredients[ing]=0
		if ing in ingredients_total:
			ingredients_total[ing]+=1
		else:
			ingredients_total[ing]=1

#sorted_ingredients = sorted(ingredients_total.items(), key=operator.itemgetter(1))

print(str(len(ingredients_total))+" ingredients loaded")

useIngredients={}

#feature selection
spice_total={}
spice_ings=[]
meat_total={}
meat_ings=[]
seafood_total={}
seafood_ings=[]
veg_total={}
veg_ings=[]
for ing in ingredients_total.keys():
	if ingredients_total.get(ing)>=0:     #use the ingredients that appared more than n times as feature
		useIngredients[ing]=0
	for word in ing.split():
		if word in spices:
			if word in spice_total:
				spice_total[word]+=1
			else:
				spice_total[word]=1
			if ing not in spice_ings:
				spice_ings.append(ing)
		if word in meat:
			if word in meat_total:
				meat_total[word]+=1
			else:
				meat_total[word]=1
			if ing not in meat_ings:
				meat_ings.append(ing)
		if word in seafood:
			if word in seafood_total:
				seafood_total[word]+=1
			else:
				seafood_total[word]=1
			if ing not in seafood_ings:
				seafood_ings.append(ing)
		if word in veg:
			if word in veg_total:
				veg_total[word]+=1
			else:
				veg_total[word]=1
			if ing not in veg_ings:
				veg_ings.append(ing)

print(str(len(spice_total))+" spices found")
print(str(len(meat_total))+" meat found")
print(str(len(seafood_total))+" seafoods found")
print(str(len(veg_total))+" vegetables found")

for dish in subdata:
	attr=useIngredients.copy()
	attr['ingUsedInDish']=0
	attr['spiceRate']=0.0
	attr['meatRate']=0.0
	attr['seafoodRate']=0.0
	attr['vegRate']=0.0
	for ing in dish['ingredients']:
		if ing in ingredients_total:
			attr[ing]+=1
			attr['ingUsedInDish']+=1
			if ing in spice_ings:
				attr['spiceRate']+=1
			if ing in meat_ings:
				attr['meatRate']+=1
			if ing in seafood_ings:
				attr['seafoodRate']+=1
			if ing in seafood_ings:
				attr['vegRate']+=1
	if attr['meatRate']+attr['seafoodRate']==0:
		attr['vegetarianDish']=1
	else:
		attr['vegetarianDish']=0
	if attr['ingUsedInDish'] > 0:
		attr['spiceRate']/=attr['ingUsedInDish']
		attr['meatRate']/=attr['ingUsedInDish']
		attr['seafoodRate']/=attr['ingUsedInDish']
		attr['vegRate']/=attr['ingUsedInDish']
	else:
		attr['spiceRate']=0
		attr['meatRate']=0
		attr['seafoodRate']=0
		attr['vegRate']=0
		print(dish['id'])
	#print(dish['id'],attr['spiceRate'],attr['ingUsedInDish'])
	X.append(attr.values())
X_ing=np.array(X)
X_extra=[]
for row in X_ing:
	X_extra.append(row[6714:6719])
X_extra=np.array(X_extra)
print("X_extra",X_extra.shape)
for dish in test:
	attr=useIngredients.copy()
	attr['ingUsedInDish']=0
	attr['spiceRate']=0.0
	attr['meatRate']=0.0
	attr['seafoodRate']=0.0
	attr['vegRate']=0.0
	for ing in dish['ingredients']:
		if ing in ingredients_total:
			attr[ing]+=1
			attr['ingUsedInDish']+=1
			if ing in spice_ings:
				attr['spiceRate']+=1
			if ing in meat_ings:
				attr['meatRate']+=1
			if ing in seafood_ings:
				attr['seafoodRate']+=1
			if ing in seafood_ings:
				attr['vegRate']+=1
	if attr['meatRate']+attr['seafoodRate']==0:
		attr['vegetarianDish']=1
	else:
		attr['vegetarianDish']=0
	if attr['ingUsedInDish'] > 0:
		attr['spiceRate']/=attr['ingUsedInDish']
		attr['meatRate']/=attr['ingUsedInDish']
		attr['seafoodRate']/=attr['ingUsedInDish']
		attr['vegRate']/=attr['ingUsedInDish']
	else:
		attr['spiceRate']=0
		attr['meatRate']=0
		attr['seafoodRate']=0
		attr['vegRate']=0
		print(dish['id'])
	#print(dish['id'],attr['spiceRate'],attr['ingUsedInDish'])
	X_unknown.append(attr.values())
X_unknown_ing=np.array(X_unknown)
X_unknown_extra=[]
for row in X_unknown_ing:
	X_unknown_extra.append(row[6714:6719])
X_unknown_extra=np.array(X_unknown_extra)
print(str(len(X[1]))+" features used")

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
tfidfts=vectorizertr.transform(corpusts).todense()

#print("tfidftr",tfidftr.shape)
#print("tfidfts",tfidfts.shape)
selpca_tfidf=PCA(n_components=2000)
selkb_tfidf=SelectKBest(f_classif,k=2000)
selp_tfidf=SelectPercentile(f_classif,percentile=95)
sel_tfidf=selp_tfidf
X_tfidf_sel = sel_tfidf.fit_transform(tfidftr,traindf['cuisine'])
X_unknown_tfidf_sel=sel_tfidf.transform(tfidfts)
print("X_tfidf_sel",X_tfidf_sel.shape)
print("X_unknown_tfidf_sel",X_unknown_tfidf_sel.shape)

#print("X_ing",X_ing.shape)
#print("X_unknown_ing",X_unknown_ing.shape)
selpca_ing=PCA(n_components=500)
selkb_ing=SelectKBest(f_classif,k=1)
selp_ing=SelectPercentile(f_classif,percentile=95)
sel_ing=selp_ing
X_ing_sel = sel_ing.fit_transform(X_ing,traindf['cuisine'])
X_unknown_ing_sel=sel_ing.transform(X_unknown_ing)
print("X_ing_sel",X_ing_sel.shape)
print("X_unknown_ing_sel",X_unknown_ing_sel.shape)

X=np.concatenate((X_tfidf_sel,X_ing_sel,X_extra),axis=1)
print("X",X.shape)
X_unknown=np.concatenate((X_unknown_tfidf_sel,X_unknown_ing_sel,X_unknown_extra),axis=1)
print("X_unknown",X_unknown.shape)
Y = traindf['cuisine']

#X_unknown = tfidfts

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'weights':[[5,4,1,1]]}
#parameters = {'n_estimators':[750],'learning_rate': [0.08],'subsample':[0.65]}
lsvc = LinearSVC()
lr = LogisticRegression(class_weight='balanced',C=10)
ovr=OneVsRestClassifier(LogisticRegression(),n_jobs=1)
rf=RandomForestClassifier(verbose=1,n_jobs=20,min_samples_leaf=1,n_estimators=500,oob_score=1,max_features='log2')
knn=KNeighborsClassifier(n_neighbors=15, algorithm = 'brute',weights='distance')
ab=AdaBoostClassifier(n_estimators=50)
bag=BaggingClassifier(n_estimators=100,max_features=0.5)
etc=ExtraTreesClassifier(verbose=1,n_jobs=20,n_estimators=1000)
mnb=MultinomialNB(alpha=0.025)
xgb=XGBClassifier()
vc=VotingClassifier(estimators=[('etc',etc),('lr', lr), ('knn', knn), ('rf', rf)], voting='soft')
classifier = grid_search.GridSearchCV(vc, parameters,verbose=2)

classifier.fit(X,Y)
#classifier.fit(X_train,Y_train)

print("Start predicting")
predictions=classifier.predict(X_unknown)
print(str(len(predictions))+" results")
print(predictions)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("submission6.csv",index = False)

print("Scoring result")
for dict in classifier.grid_scores_:
    print(dict)

print("Train set accuracy: "+str(classifier.score(X_train,Y_train)))
print("Test set accuracy: "+str(classifier.score(X_test,Y_test)))