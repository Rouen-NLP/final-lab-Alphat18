#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:24:09 2018

@author: barbith3
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

dictionary = {}
classes = {}
nb_word_label = {}

#%%

for root, dirs, files in os.walk("data/Tobacco3482-OCR/"):
    for file in files:
        if file.endswith(".txt"):
            if root.split('/')[2] in dictionary:
                dictionary[root.split('/')[2]].append(os.path.join(root, file))
                classes[root.split('/')[2]] += 1
            else:
                dictionary[root.split('/')[2]] = []
                classes[root.split('/')[2]] = 0

fig = plt.figure()
plt.bar(list(classes.keys()), list(classes.values()))

#%%

vector_text = []
vector_category = []

for label in classes.keys():
    nb_word_label[label] = 0
    for file in dictionary[label]:
        with open(file) as f:
            
            vector_text.append(f.read())
            vector_category.append(label)
            
            words = [word for line in f for word in line.split()]   
            nb_word_label[label] += len(words)
    nb_word_label[label] /= classes[label]

fig = plt.figure()
plt.bar(list(nb_word_label.keys()), list(nb_word_label.values()))

#%%
from sklearn.model_selection import train_test_split
# Split the dataset, create X (features) and y (target), print the size
X_train, X_test, y_train, y_test = train_test_split(vector_text, vector_category, test_size=0.2)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
# Create document vectors
vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_dev_counts = vectorizer.transform(X_dev)
X_test_counts = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
# train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
y_hat_train = clf.predict(X_train_counts)

error_train = len(y_hat_train[y_hat_train == y_train]) / len(y_train)

print(error_train)

y_hat_dev = clf.predict(X_dev_counts)
error_dev = len(y_hat_dev[y_hat_dev == y_dev]) / len(y_dev)

print(error_dev)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

report = classification_report(y_dev, y_hat_dev)
print(report)
confusion = confusion_matrix(y_dev, y_hat_dev)
plt.imshow(confusion)
plt.colorbar()