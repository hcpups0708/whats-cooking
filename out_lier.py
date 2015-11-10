__author__ = 'Asus Huang'
import json
import math
import io
import random
import multiprocessing as mp
import re
import os
import operator
import numpy as np
from numpy.linalg import  norm
from pprint import pprint
from sklearn.preprocessing import StandardScaler
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
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors

def main():
    country={}
    ingredients={}
    ingredients_total={}
    X=[]
    Y=[]
    X_unknown=[]
    #gnb=GaussianNB()
    gnb=OneVsRestClassifier(LogisticRegression(),n_jobs=1)
    #gnb=OneVsOneClassifier(LinearSVC(random_state=0))
    #gnb = AdaBoostClassifier(n_estimators=50)
    #gnb=GradientBoostingClassifier(verbose=2)
    #gnb=NearestCentroid(metric='euclidean')
    #gnb = KNeighborsClassifier(n_neighbors=1, algorithm = 'auto')
    #gnb=RandomForestClassifier(verbose=1,n_jobs=2,min_samples_leaf=1,n_estimators=200,oob_score=1)

    #load from json file
    with io.open('train_cleaned.json', encoding = 'utf8') as data_file:
        data = json.load(data_file)

    with io.open('test.json', encoding = 'utf8') as data_file:
        test = json.load(data_file)

    with io.open('spice.txt', encoding = 'utf8') as f:
        spices = f.read().splitlines()



    #subdata=data                        #use full data for training
    subdata=random.sample(data,1000)   #randomly select n data for training


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

    print(str(len(spice_total))+" spices found")

    for dish in subdata:
        attr=useIngredients.copy()
        attr['ingUsedInDish']=0
        attr['spiceRate']=0.0
        for ing in dish['ingredients']:
            if ing in ingredients_total:
                attr[ing]+=1
                attr['ingUsedInDish']+=1
                if ing in spice_ings:
                    attr['spiceRate']+=1
        if attr['ingUsedInDish'] > 0:
            attr['spiceRate']/=attr['ingUsedInDish']
        else:
            attr['spiceRate']=0
        #print(dish['id'],attr['spiceRate'],attr['ingUsedInDish'])
        X.append(attr.values())
        Y.append(dish['cuisine'])

    for dish in test:
        attr=useIngredients.copy()
        attr['ingUsedInDish']=0
        attr['spiceRate']=0.0
        for ing in dish['ingredients']:
            if ing in ingredients_total:
                attr[ing]+=1
                attr['ingUsedInDish']+=1
                if ing in spice_ings:
                    attr['spiceRate']+=1
        if attr['ingUsedInDish'] > 0:
            attr['spiceRate']/=attr['ingUsedInDish']
        else:
            attr['spiceRate']=0
        #print(dish['id'],attr['spiceRate'],attr['ingUsedInDish'])
        X_unknown.append(attr.values())
    print(str(len(X[1]))+" features used")
    #X = StandardScaler().fit_transform(X)
    #neigh = NearestNeighbors(radius=1000)
    #neigh.fit(X)
    #rng = neigh.radius_neighbors(X[0])
    #for group in rng[0]:
    #    print(group)
    A = {}
    for c in country:
        A[c] = []
        for idx,data in enumerate(X):
            if  subdata[idx]['cuisine'] == c:
                A[c].append(data)

    AP={}
    for c in country:
        AP[c]=[]
        AP[c] = [0] * len(A[c][0])
        for p in A[c]:
            AP[c] = [sum(x) for x in zip(AP[c], p)]

#average-mean
    for c in country:
        myInt = len(A[c])*1.0
        AP[c] = [x / myInt for x in AP[c]]


#Distance
    mean_Distance={}
    Distance={}

    for c in country:
        Distance[c]=[]
        for p in A[c]:
            val=norm(np.array(AP[c])-np.array(p))
            Distance[c].append(val)
        mean_Distance[c] = sum(Distance[c])/float(len(Distance[c]))

#Distance-Standard
    ST={}
    for c in country:
        ST[c]=np.std(Distance[c])
#filter
    subdata_copy=list(subdata)
    del subdata[:]
    for ind,data in enumerate(X):
        c=subdata_copy[ind]['cuisine']
        if (norm(np.array(data)-np.array(AP[c]))-mean_Distance[c])/ST[c] < 2:
            subdata.append(subdata_copy[ind])
    del subdata_copy

    with io.open('train_without_outlier.json', 'wb') as output_file:
        json.dump(subdata,output_file, indent=2)
    print('123')
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
"""
    print("Start training")
    #os.system("pause")

    gnb.fit(X_train,Y_train)
    #gnb.fit(X,Y)

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

    print("Scoring result")

    print("Train set accuracy: "+str(gnb.score(X_train,Y_train)))
    print("Test set accuracy: "+str(gnb.score(X_test,Y_test)))
    #pprint(gnb.staged_score(X,Y)) #Only for AdaBoost

    #pprint(data)
"""
if __name__=='__main__':
    mp.freeze_support()
    main()