__author__ = 'Asus Huang'
import json
import io
import random
import operator
from pprint import pprint
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
X_unknown=[]
#gnb=GaussianNB()
#gnb=OneVsRestClassifier(LinearSVC(random_state=0))
#gnb=OneVsOneClassifier(LinearSVC(random_state=0))
#gnb = AdaBoostClassifier(n_estimators=50)
#gnb=GradientBoostingClassifier(verbose=2)
#gnb=NearestCentroid(metric='euclidean')
#gnb = KNeighborsClassifier(n_neighbors=1, algorithm = 'auto')
gnb=RandomForestClassifier(verbose=1,n_jobs=20,min_samples_leaf=1,n_estimators=1000,oob_score=1)

#load from json file
with io.open('train.json', encoding = 'utf8') as data_file:
    data = json.load(data_file)

with io.open('test.json', encoding = 'utf8') as data_file:
    test = json.load(data_file)



subdata=data                        #use full data for training
#subdata=random.sample(data,20000)   #randomly select n data for training

#count and add ingredients
for dish in subdata:
    country[dish['cuisine']]=0
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
for ing in ingredients_total.keys():
    if ingredients_total.get(ing)>=0:     #use the ingredients that appared more than n times as feature
        useIngredients[ing]=0

print(str(len(useIngredients))+" feature used")

for dish in subdata:
    attr=useIngredients.copy()
    country[dish['cuisine']]+=1;
    for ing in dish['ingredients']:
        if ing in ingredients_total:
            attr[ing]+=1
    X.append(attr.values())
    Y.append(dish['cuisine'])

for dish in test:
    attr=useIngredients.copy()
    for ing in dish['ingredients']:
        if ing in ingredients_total:
            attr[ing]+=1
    X_unknown.append(attr.values())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("Start training")

#gnb.partial_fit(X,Y,country.keys()) #Only for GaussianNB
gnb.fit(X_train,Y_train)

print("Start predicting")

result=gnb.predict(X_unknown)

print("Output result")

out=open("ans.csv",'w')
out.write("id,cuisine\n")

i=0

for dish in test:
    out.write(str(dish['id'])+","+result[i]+"\n")
    i+=1


print(result)

print("Scoring result")

print("Train set accuracy: "+str(gnb.score(X,Y)))
print("Test set accuracy: "+str(gnb.score(X_test,Y_test)))
#pprint(gnb.staged_score(X,Y)) #Only for AdaBoost

#pprint(data)