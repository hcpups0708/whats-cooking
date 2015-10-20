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
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC

country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
X_test=[]
#gnb=GaussianNB()
#gnb=OneVsRestClassifier(LinearSVC(random_state=0))
#gnb=OneVsOneClassifier(LinearSVC(random_state=0))
#gnb = AdaBoostClassifier(n_estimators=50)
gnb=GradientBoostingClassifier(verbose=1)
#gnb=NearestCentroid(metric='euclidean')


with io.open('train.json', encoding = 'utf8') as data_file:
    data = json.load(data_file)

with io.open('test.json', encoding = 'utf8') as data_file:
    test = json.load(data_file)



subdata=data
#subdata=random.sample(data,13000)

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

for ing in ingredients_total.keys():
    if ingredients_total.get(ing)>=0:
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
    X_test.append(attr.values())

print("Start training")

#gnb.partial_fit(X,Y,country.keys()) #Only for GaussianNB
gnb.fit(X,Y)

print("Start predicting")

result=gnb.predict(X_test)

print("Output result")

out=open("ans.csv",'w')
out.write("id,cuisine\n")

i=0

for dish in test:
    out.write(str(dish['id'])+","+result[i]+"\n")
    i+=1


print(result)

print("Scoring result")

print(gnb.score(X,Y))
#pprint(gnb.staged_score(X,Y)) #Only for AdaBoost

#pprint(data)