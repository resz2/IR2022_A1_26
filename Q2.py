#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import os
from collections import defaultdict
import copy
from functools import reduce


# In[2]:


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


# In[3]:


len(docnames)


# In[ ]:





# ## A: Preprocessing

# In[4]:


stop_words = set(stopwords.words('english'))


# In[ ]:





# In[5]:


def preprocess(doc):
    procd = []
    for line in doc:
        line = line.lower()
        line = word_tokenize(line)
        tokens = []
        for word in line:
            if(word not in stop_words and word.isalnum()):
                procd.append(word)
    return procd


# In[6]:


prodocs = []

for doc in docs:
    procd = preprocess(doc)
    prodocs.append(procd)


# In[ ]:





# ## B: Positional Index

# In[7]:


index = defaultdict(list)

for i, doc in enumerate(prodocs):
    for j, word in enumerate(doc):
        if word in index:
            index[word][0] += 1
            if(i in index[word][1]):
                index[word][1][i].append(j)
            else:
                index[word][1][i] = [j]
        else:
            index[word] = []
            index[word].append(1)
            index[word].append({})
            index[word][1][i] = [j]


# In[8]:


index['rainbow']


# In[ ]:





# ## C: Phrase Queries

# In[9]:


def intersect1(l1, l2):
    return list(set(l1).intersection(l2))

def intersector(lists):
    lists.sort(key=len)
    return list(reduce(intersect1, lists))

def process(query_tokens, prodocs):
    ids = []
    for token in query_tokens:
        if token not in index:
            return []

    postings = [index[token] for token in query_tokens]
    tokendocs = [p[1].keys() for p in postings]
    # getting docs that contain all query terms
    docalltokens = intersector(tokendocs)
    
    valid_docs = []
    for docid in docalltokens:
        positions = []
        for p in postings:
            positions.append(p[1][docid])
        copied = copy.deepcopy(positions)
        
        # checking correct order
        for i in range(len(copied)):
            for j in range(len(copied[i])):
                copied[i][j] -= i

        # check intersect to see if contain phrase
        combined = intersector(copied)
        if(not combined):
            continue
        else:
            valid_docs.append(docid)
    return sorted(valid_docs)


# In[ ]:





# ## D: Testing

# In[10]:


query = 'good morning'
tokens = preprocess([query])
tokens


# In[11]:


doclist = process(tokens, prodocs)


# In[12]:


if(not doclist):
    print('No document was dound to contain the given phrase')
else:
    print('Number of documents:', len(doclist))
    print()
    for id in doclist:
        print(docnames[id])


# In[13]:


input_query = input('Enter the query: ')
input_tokens = preprocess([input_query])
result = process(input_tokens, prodocs)


# In[14]:


if(not result):
    print('No document was dound to contain the given phrase')
else:
    print('Number of documents:', len(result))
    print()
    for docid in result:
        print(docnames[docid])


# In[ ]:





# In[ ]:




