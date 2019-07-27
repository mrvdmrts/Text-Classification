# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 00:38:38 2019

@author: Merve
"""

import pandas as pd
import numpy as np
import os
import locale
locale.setlocale(locale.LC_ALL,"")

headers=["kategori","haber"]
#%%   Test ve eğitim verilerinin okunması

data_train=[]
data_test=[]

def veri_okuma(tip,sayi):
    data=[]
    kategori=os.listdir(os.getcwd()+"/odev-veriler/"+tip)
    for i in kategori:
        for j in os.listdir(os.getcwd()+"/odev-veriler/"+tip+"/"+i):
            text=open(os.getcwd()+"/odev-veriler/"+tip+"/"+i+"/"+j,"r")
            data+=[i,text.read()]
    np_data=np.array(data)
    np_data=np_data.reshape(sayi,2)
    df=pd.DataFrame(np_data,columns=headers)
    data=df
    return data

data_train=veri_okuma("train",600)
data_test=veri_okuma("test",320)

#%%Cleaning Data

import re  #Regex

words= open(os.getcwd()+'/stop-words-turkish-github.txt')
stop_words=[line.strip() for line in open(os.getcwd()+'/stop-words-turkish-github.txt')]
stop_words=np.array(stop_words)
#%%
#Kök Çözümleme
from snowballstemmer import stemmer

kokbul = stemmer('turkish')

#%% Test ve eğitim verileri için veri temizleme işlemleri
def veri_temizle(data):
    haber_list=[]
    for haber in data.haber:
        haber=haber.lower()
        haber=re.sub("[^abcçdefgğhıijklmnoöprsştuüvyzwxq ]","",haber)
        haber=haber.split(" ")
        haber=[word for word in haber if not word in set(stop_words)]  
        #haber=kokbul.stemWords(haber)
        #haber=" ".join(haber)
        haber_list.append(haber)
    return haber_list

haber_list_train=veri_temizle(data_train)
haber_list_test=veri_temizle(data_test)

#%%
def kok_bulma(data):
    haber=[]
    for x in data:
        x=kokbul.stemWords(x)
        x=" ".join(x)
        haber.append(x)
    return haber
haber_list_train=kok_bulma(haber_list_train)
haber_list_test=kok_bulma(haber_list_test)
#%%
for i in range(600):
    data_train.haber[i]=haber_list_train[i]
for i in range(320):
    data_test.haber[i]=haber_list_test[i]
#%%
    X_train = data_train['haber']
    y_train = data_train['kategori']
    X_test = data_test['haber']
    y_test = data_test['kategori']
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=3, sublinear_tf=True, norm='l2', ngram_range=(1, 6))
final_features = vectorizer.fit_transform(data_train['haber']).toarray()

#%%
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', RandomForestClassifier(max_features='auto',max_depth=30))])
    
model = pipeline.fit(X_train, y_train)
ytest = np.array(y_test)   
#%%
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', MultinomialNB())])
    
model = pipeline.fit(X_train, y_train)
ytest = np.array(y_test)
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

acc = accuracy_score(model.predict(X_test), ytest)
recall=recall_score(model.predict(X_test),ytest,average='macro')
precision=precision_score(model.predict(X_test),ytest,average='micro')
f1=f1_score(model.predict(X_test),ytest,average='micro')


#%%

# Evaluation
report=classification_report(ytest, model.predict(X_test))
print(confusion_matrix(ytest, model.predict(X_test)))

matrix=confusion_matrix(ytest, model.predict(X_test))
#%%


print("ACCURACY="+ str(acc))
print("RECALL_SCORE=" + str(recall))
print("PRECİSİON="+str(precision))
print("F SCORE="+str(f1))