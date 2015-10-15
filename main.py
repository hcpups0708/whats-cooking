__author__ = 'Asus Huang'
import json
import random
import operator
from pprint import pprint
from sklearn.naive_bayes import GaussianNB

country={}
ingredients={}
ingredients_total={}
X=[]
Y=[]
X_test=[]
gnb=GaussianNB()

with open('train.json') as data_file:
    data = json.load(data_file)

with open('test.json') as data_file:
    test = json.load(data_file)



subdata=data
#subdata=random.sample(data,21000)

for dish in subdata:
    country[dish['cuisine']]=0
    for ing in dish['ingredients']:
        ingredients[ing]=0
        if ingredients_total.has_key(ing):
            ingredients_total[ing]+=1
        else:
            ingredients_total[ing]=0

sorted_ingredients = sorted(ingredients_total.items(), key=operator.itemgetter(1))
#pprint(sorted_ingredients)

useIngredients={}

for ing in ingredients_total.keys():
    if ingredients_total.get(ing)>10:
        useIngredients[ing]=0

print(len(useIngredients))

for dish in subdata:
    attr=useIngredients.copy()
    country[dish['cuisine']]+=1;
    for ing in dish['ingredients']:
        if useIngredients.has_key(ing):
            attr[ing]+=1
    X.append(attr.values())
    Y.append(dish['cuisine'])

for dish in test:
    attr=useIngredients.copy()
    for ing in dish['ingredients']:
        if useIngredients.has_key(ing):
            attr[ing]+=1
    X_test.append(attr.values())


gnb.partial_fit(X,Y,country.keys())
result=gnb.predict(X_test)
print(result)
#print(gnb.score(X,Y))

out=open("ans.csv",'w')
out.write("id,cuisine\n")

i=0

for dish in test:
    out.write(str(dish['id'])+","+result[i]+"\n")
    i+=1


#pprint(data)