__author__ = 'Asus Huang'
import json
import io
import random
import operator
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC

country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
X_test=[]


label='data=5000,n_estimators=20,learning_rate=0.1,subsample=1'
#gnb=GaussianNB()
#gnb=OneVsRestClassifier(LinearSVC(random_state=0))
#gnb=OneVsOneClassifier(LinearSVC(random_state=0))
#gnb = AdaBoostClassifier(n_estimators=50)
n_estimators=20
gnb=GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=0.1,subsample=1,verbose=2)
#gnb=NearestCentroid(metric='euclidean')
#gnb = KNeighborsClassifier(n_neighbors=1, algorithm = 'auto')

#load from json file
with io.open('train.json', encoding = 'utf8') as data_file:
    data = json.load(data_file)

with io.open('test.json', encoding = 'utf8') as data_file:
    test = json.load(data_file)



#subdata=data                        #use full data for training
subdata=random.sample(data,5000)   #randomly select n data for training

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

print(str(len(result))+" results")
print(result)


print("Plotinging result")
# compute train set deviance
train_deviance = np.zeros((n_estimators,), dtype=np.float64)

npY=np.array(Y)
for i, y_pred in enumerate(gnb.staged_decision_function(X)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    train_deviance[i] = gnb.loss_(npY, y_pred)

plt.plot((np.arange(train_deviance.shape[0]) + 1)[::5], train_deviance[::5], '-', color='blue', label=label)

plt.legend(loc='upper left')
plt.xlabel('Boosting Iterations')
plt.ylabel('Train Set Deviance')

plt.show()

print("Scoring result")
print(gnb.score(X,Y))
#pprint(gnb.staged_score(X,Y)) #Only for AdaBoost

#pprint(data)