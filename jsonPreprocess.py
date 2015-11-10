__author__ = 'Asus Huang'
import json
import io
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

with io.open('train.json', encoding = 'utf8') as data_file:
        data = json.load(data_file)
with io.open('test.json', encoding = 'utf8') as test_file:
        test = json.load(test_file)

for idx, dish in enumerate(data):
    for idx2, ing in enumerate(dish['ingredients']):
        words=[]
        for w in [WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', '', word)) for word in re.sub("[^\w]", " ", ing).split()]:
            #if w not in stopwords.words('english'):
                words.append(w.lower())
        data[idx]['ingredients'][idx2]=' '.join(words)
    #print(idx)

for idx, dish in enumerate(test):
    for idx2, ing in enumerate(dish['ingredients']):
        words=[]
        for w in [WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', '', word)) for word in re.sub("[^\w]", " ", ing).split()]:
            #if w not in stopwords.words('english'):
                words.append(w.lower())
        test[idx]['ingredients'][idx2]=' '.join(words)

with io.open('train_cleaned_with_stop.json', 'wb') as data_file:
        json.dump(data,data_file, indent=2)

with io.open('test_cleaned_with_stop.json', 'wb') as test_file:
        json.dump(test,test_file, indent=2)

#outString=json.dumps(data, indent=4)

#out=io.open("train_cleaned.json",'wb')
#out.write(outString)


