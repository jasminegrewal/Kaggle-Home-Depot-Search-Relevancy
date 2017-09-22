# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:22:26 2016

@author: jasmine
"""
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import collections
import math,re
from numpy import ones, zeros
import datetime
from collections import Counter

start = datetime.datetime.now()
#reading all csv files
train = pd.read_csv('train.csv',encoding="ISO-8859-1")
test = pd.read_csv('test.csv',encoding="ISO-8859-1")
attrs= pd.read_csv('attributes.csv',encoding="ISO-8859-1")
prdesr=pd.read_csv('product_descriptions.csv',encoding="ISO-8859-1")

#extracting the data from attributes file which I want to include for the model
brand = attrs[attrs.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
bullet1 = attrs[attrs.name == "Bullet01"][["product_uid", "value"]].rename(columns={"value": "bullet1"})
bullet2 = attrs[attrs.name == "Bullet02"][["product_uid", "value"]].rename(columns={"value": "bullet2"})
bullet3 = attrs[attrs.name == "Bullet03"][["product_uid", "value"]].rename(columns={"value": "bullet3"})
Material = attrs[attrs.name == "Material"][["product_uid", "value"]].rename(columns={"value": "Material"})

#merging the extracting data as columns into one 
train = pd.merge(train, brand, how='left', on='product_uid')
train = pd.merge(train, bullet1, how='left', on='product_uid')
train = pd.merge(train, bullet2, how='left', on='product_uid')
train = pd.merge(train, bullet3, how='left', on='product_uid')
train = pd.merge(train, Material, how='left', on='product_uid')
train = pd.merge(train, prdesr, how='left', on='product_uid')

#train.to_csv("train.csv", index=False)

#defining a lambda function to execute on the columns in the training data which removes stopwords and performs stemming
lowerize = lambda x: stmstp(x)

def stmstp(s):
    if isinstance(s, str):
        s=s.lower()
        p=[]
        p=s.split()
        sw=sorted(stopwords.words('english'))
        p=[i for i in p if i not in sw ]
        stemmer = PorterStemmer()
        s=[stemmer.stem(i) for i in p]
        s= " ".join(s)
    return s

#applying the function on all columns
train.search_term = train.search_term.apply(lowerize)
train.product_title = train.product_title.apply(lowerize)
train.product_description = train.product_description.apply(lowerize)
train.brand = train.brand.apply(lowerize)
train.bullet1 = train.bullet1.apply(lowerize)
train.bullet2 = train.bullet2.apply(lowerize)
train.bullet3 = train.bullet3.apply(lowerize)
train.Material = train.Material.apply(lowerize)
       
# dividing training data into two parts: data and query
#data has concatenated data of columns: product title, product description, brand,material, bullet1, bullet2 and bullet3
#query has the data from column search term 
#below I am calculating the cosine similarity between data and query based only on term frequency
#data dictionary has key as sequence-wise generated integers and value as data, same with query and hence they have data from same row 

data={}
query={}

i=0
for index,row in train.iterrows():
    keys=list(row.keys())
    product_id=i
    i +=1
    query[product_id]=row['search_term']
    keys.remove('product_uid')
    keys.remove('search_term')
    keys.remove('id')
    keys.remove('relevance')
    st=""
    for k in keys:
        if isinstance(row[k],str):
            st=st+row[k]
        else:
            st=""
    data[product_id]=st


datatf=collections.Counter()     
srctf=collections.Counter() 

def calctf(s,pid):
    datatf[pid]={}
    if isinstance(s,str):
        p=[] 
        p=s.split()
        for word in p:
            if word in datatf[pid]:
                datatf[pid][word]=datatf[pid][word]+1  
            else:
                datatf[pid][word]=1

def calctf2(s,pid):
    srctf[pid]={}
    if isinstance(s,str):
        p=[] 
        p=s.split()
        for word in p:
            if word in srctf[pid]:
                srctf[pid][word]=srctf[pid][word]+1  
            else:
                srctf[pid][word]=1               
for k in data:
    calctf(data[k],k)
    
for k in query:
    calctf2(query[k],k)

k=list(srctf.keys())

def cosine_sim(str1, str2):
     common = set(str1.keys()) & set(str2.keys())
     prod = sum([str1[x] * str2[x] for x in common])

     sum1 = sum([str1[x]**2 for x in str1.keys()])
     sum2 = sum([str2[x]**2 for x in str2.keys()])
     div = math.sqrt(sum1) * math.sqrt(sum2)

     if not div:
        return 0.0
     else:
        return float(prod) / div

sim={}
for keys in k:
    sim[keys]=cosine_sim(datatf[keys],srctf[keys])
    
#Linear Regression
#implementing Linear Regression on training data where x is calculated 
    #similarity values and y is the relevnce values
x = np.fromiter(iter(sim.values()), dtype=float)
y=train['relevance'].values

m=y.size
X = ones(shape=(m, 2))
X[:, 1] = x

theta = zeros(shape=(2, 1))
iterations = 2000
alpha = 0.01    

def calc_cost(X, y, theta):
   
    m = y.size

    new_x = X.dot(theta).flatten()

    sqrerr = (new_x - y) ** 2

    J = (1.0 / (2 * m)) * sqrerr.sum()

    return J

def gradient_descent(X, y, theta, alpha, iters):
    
    m = y.size
    cost = zeros(shape=(iters, 1))

    for i in range(iters):

        new_x = X.dot(theta).flatten()

        err_x1 = (new_x - y) * X[:, 0]
        err_x2 = (new_x - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * err_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * err_x2.sum()

        cost[i, 0] = calc_cost(X, y, theta)

    return theta, cost
#get theta values from linear regression algorithms and cost values are just to check if cost is reducing
theta , cost = gradient_descent(X, y, theta, alpha, iterations)         

#merging data for test same way as train and applying the lowerize function
test = pd.merge(test, brand, how='left', on='product_uid')
test = pd.merge(test, bullet1, how='left', on='product_uid')
test = pd.merge(test, bullet2, how='left', on='product_uid')
test = pd.merge(test, bullet3, how='left', on='product_uid')
test = pd.merge(test, Material, how='left', on='product_uid')
test = pd.merge(test, prdesr, how='left', on='product_uid')

test.search_term = test.search_term.apply(lowerize)
test.product_title = test.product_title.apply(lowerize)
test.product_description = test.product_description.apply(lowerize)
test.brand = test.brand.apply(lowerize)
test.bullet1 = test.bullet1.apply(lowerize)
test.bullet2 = test.bullet2.apply(lowerize)
test.bullet3 = test.bullet3.apply(lowerize)
test.Material = test.Material.apply(lowerize)

#function to find and count words
WORD = re.compile(r'\w+')
def count_words(text):
     words = WORD.findall(text)
     return Counter(words)

output={}
pid=[]
relevance=[]
d=dict()
#dividing the test data same way as train search term and other attributes
for index,row in test.iterrows():
    keys=list(row.keys())
    product_id=row['id']
    text2=row['search_term']
    keys.remove('product_uid')
    keys.remove('search_term')
    keys.remove('id')
    st=""
    for k in keys:
        if isinstance(row[k],str):
            st=st+row[k]
    
    text1=st
    vector1 = count_words(text1)
    vector2 = count_words(text2)
    
#calculating the similarity between two
    x1 = cosine_sim(vector1, vector2)
#predicting relevance using the output of the linear regression
    y1= round(float(theta[0])+(float(theta[1])*x1),3)
    
    if y1<0:
        y1=1
    if y1>3:
        y1=3
#in pid storing the product_uid from the test data and relevance has predicted relevance   
    if product_id not in d:
        pid.append(product_id)
        d[product_id]=1
        relevance.append(y1)

#appending both the lists into output file
output['id']=pid
output['relevance']=relevance

#making dataframe from output dictionary and writing it into csv file    
df = pd.DataFrame(output, columns = ['id', 'relevance'])   
df.to_csv('result.csv', index=False)
end = datetime.datetime.now()
time_taken= end-start
print ('time to run program is: ', time_taken)