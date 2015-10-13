__author__ = 'Asus Huang'
import json
import random
from pprint import pprint
from sklearn.naive_bayes import GaussianNB

country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
gnb=GaussianNB()

with open('train.json') as data_file:
    data = json.load(data_file)

subdata=random.sample(data,10000)

for dish in subdata:
    country[dish['cuisine']]=0
    for ing in dish['ingredients']:
        ingredients[ing]=0
        ingredients_total[ing]=0

for dish in subdata:
    attr=ingredients.copy()
    country[dish['cuisine']]+=1;
    for ing in dish['ingredients']:
        attr[ing]+=1
        ingredients_total[ing]+=1
    X.append(attr.values())
    Y.append(dish['cuisine'])

gnb.partial_fit(X,Y,country.keys())
print(gnb.predict(X))
print(gnb.score(X,Y))


#pprint(data)