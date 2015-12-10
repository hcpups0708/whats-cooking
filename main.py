__author__ = 'Asus Huang'
import json
import io
import random
import multiprocessing as mp
import re
import os
import operator
import numpy as np
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

def main():
    country={}
    ingredients={}
    ingredients_total={}
    X=[]
    Y=[]
    Y_v_t=[]
    X_unknown=[]
    #gnb=GaussianNB()
    ovr=OneVsRestClassifier(LogisticRegression(),n_jobs=1)
    ovo=OneVsOneClassifier(LogisticRegression())
    rf=RandomForestClassifier(verbose=1,n_jobs=20,min_samples_leaf=1,n_estimators=300,oob_score=1)
    knn=KNeighborsClassifier(n_neighbors=5, algorithm = 'auto')
    #gnb=OneVsRestClassifier(LogisticRegression(),n_jobs=1)
    #gnb=OneVsOneClassifier(LinearSVC(random_state=0))
    #gnb = AdaBoostClassifier(n_estimators=50)
    #gnb=GradientBoostingClassifier(verbose=2)
    #gnb=NearestCentroid(metric='euclidean')
    gnb = KNeighborsClassifier(n_neighbors=1, algorithm = 'auto', n_jobs=2)
    #gnb=RandomForestClassifier(verbose=1,n_jobs=2,min_samples_leaf=1,n_estimators=200,oob_score=1)
    #gnb=VotingClassifier(estimators=[('ovr', ovr), ('knn', knn), ('rf', rf)], voting='soft', weights=[3,1,2])

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
    #subdata=random.sample(data,2000)   #randomly select n data for training


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
        Y.append(dish['cuisine'])
        if dish['cuisine']=='vietnamese' or dish['cuisine']=='thai':
            Y_v_t.append(1)
        else:
            Y_v_t.append(0)
    X=np.array(X)
    Y=np.array(Y)

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
    X_unknown=np.array(X_unknown)

    print(str(len(X[1]))+" features used")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    print("Start training")
    #os.system("pause")

    #gnb.fit(X_train,Y_train)
    gnb.fit(X,Y)

    #print(gnb.feature_importances_)

    print("Start predicting")

    result=gnb.predict(X_unknown)

    print("Output result")

    out=open("ans.csv",'w')
    out.write("id,cuisine\n")

    i=0

    for dish in test:
        out.write(str(dish['id'])+","+result[i]+"\n")
        i+=1

    print(str(len(result))+" results")
    print(result)
    out.close()

    print("Scoring result")

    print("Train set accuracy: "+str(gnb.score(X_train,Y_train)))
    print("Test set accuracy: "+str(gnb.score(X_test,Y_test)))
    #pprint(gnb.staged_score(X,Y)) #Only for AdaBoost

    #pprint(data)

if __name__=='__main__':
    mp.freeze_support()
    main()