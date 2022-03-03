#!/usr/bin/env python
# coding: utf-8

# In[74]:


import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import os
from collections import defaultdict
import copy
from functools import reduce
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[75]:


docs = []
docnames = []

for filename in os.listdir('data'):
    try:
        lines = []
        file = open('data/'+filename, 'r')
        for i in file:
            lines.append(i.replace('\n', ''))
        docnames.append(filename)
        docs.append(lines)
    except:
        print('File not read:', filename)


# In[76]:


len(docnames)


# In[ ]:





# ## A: Preprocessing

# In[77]:


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[107]:


def preprocess(doc):
    procd = []
    for line in doc:
        line = line.lower()
        line = word_tokenize(line)
        tokens = []
        for word in line:
            if(word not in stop_words and word.isalnum()):
                procd.append(stemmer.stem(word))
                #procd.append(lemmatizer.lemmatize(word))
                #procd.append(word)
    return procd


# In[108]:


prodocs = []

for doc in docs:
    procd = preprocess(doc)
    prodocs.append(procd)


# In[ ]:





# ## B: Unigram Inverted Index

# In[109]:


index = defaultdict(list)

for i, doc in enumerate(prodocs):
    for j, word in enumerate(doc):
        if word in index:
            if i not in index[word][1]:
                index[word][1].append(i)
            index[word][0] = len(index[word][1])
        else:
            index[word] = [1]
            index[word].append([i])


# In[110]:


index['rainbow']


# In[ ]:





# ## C: Queries Handling

# In[111]:


def ander(l1, l2):
    if(len(l2) < len(l1)):
        l1, l2 = l2, l1
    ret = []
    comparisons = 0
    i = j = 0
    
    while(i<len(l1) and j<len(l2)):
        if(l1[i] == l2[j]):
            ret.append(l1[i])
            i += 1
            j += 1
        elif(l1[i] < l2[j]):
            i += 1
        else:
            j += 1
        comparisons += 1
    
    return comparisons, ret

def orer(l1, l2):
    ret = []
    comparisons = 0
    i = j = 0
    
    while(i<len(l1) and j<len(l2)):
        if(l1[i] == l2[j]):
            ret.append(l1[i])
            i += 1
            j += 1
        elif(l1[i] < l2[j]):
            ret.append(l1[i])
            i += 1
        else:
            ret.append(l2[j])
            j += 1
        comparisons += 1
    
    while(i<len(l1)):
        ret.append(l1[i])
        i += 1
    while(j<len(l2)):
        ret.append(l2[j])
        j += 1
    
    return comparisons, ret

def noter(l1):
    idlist = list(range(len(prodocs)))
    for docid in l1:
        idlist.remove(docid)
    return idlist


# In[124]:


def handle(l1, l2, op):
    if(op == 'and'):
        return ander(l1, l2)
    elif(op == 'or'):
        return orer(l1, l2)
    elif(op == 'or not'):
        return orer(l1, noter(l2))
    elif(op == 'and not'):
        return ander(l1, noter(l2))
    else:
        print('Invalid operation')
        return -1, []

def process(query, ops):
    tokens = preprocess([query])
    ops = [op.lower().strip() for op in ops]
    print('Query tokens:', tokens)
    print('Operations:', ops)
    comparisons = 0
    
    # Input is assumed to be in correct format
    if(tokens and ops):
        l1 = index[tokens[0]][1]
        for i in range(len(ops)):
            l2 = index[tokens[i+1]][1]
            numc, l1 = handle(l1, l2, ops[i])
            if(numc == -1):
                return
            comparisons += numc
        
        return comparisons, l1
    else:
        print('Empty input')


# In[ ]:





# In[ ]:





# ## D: Testing

# In[126]:


query = 'lion stood thoughtfully for a moment'
ops = ['or', 'or', 'or']
ops1 = ['or', 'and', 'or not']
ops2 = ['or', 'or', 'and not']
ops3 = ['or', 'and', 'or not']
ops4 = ['or', 'or', 'or not']


# In[127]:


a = process(query, ops)
print(a[0], len(a[1]))


# In[128]:


a = process(query, ops1)
print(a[0], len(a[1]))


# In[129]:


a = process(query, ops2)
print(a[0], len(a[1]))


# In[130]:


a = process(query, ops3)
print(a[0], len(a[1]))


# In[131]:


a = process(query, ops4)
print(a[0], len(a[1]))


# In[ ]:





# In[132]:


input_query = input('Enter the query: ')
print('\nEnter operations separated by ",",')
input_ops = list(map(str,input("\nEnter the operations : ").strip().split(',')))


# In[136]:


result = process(input_query, input_ops)
print('\nNumber of documents matched:', len(result[1]))
print('Number of comparisons:', result[0])
print('\nDocument list\n')

for docid in result[1]:
    print(docnames[docid])


# In[ ]:





# In[ ]:




