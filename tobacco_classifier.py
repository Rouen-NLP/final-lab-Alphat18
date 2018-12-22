
# coding: utf-8

# # Classification des documents du procès des groupes américains du tabac

from time import time
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from pprint import pprint
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def load_texts(path: str) -> (list, list):
    texts = []
    categories = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                with open(root + '/' + file, 'r', encoding="utf8") as text:
                    texts.append(text.read())
                    categories.append(root.split('/')[2])
    return texts, categories


def create_data_frame(texts: list, categories: list) -> pd.DataFrame:
     return pd.DataFrame({'text': texts, 'category': categories})


def plot_category_histogram(df: pd.DataFrame):
    plt.figure(figsize=(15, 5))
    plt.title("Number of texts by categories")
    sns.countplot(data=df, x='category')
    plt.show()

def print_average_number_words(df: pd.DataFrame):
    categories = np.unique(df.category)
    word_averages = []
    for category in categories:
        word_averages.append(np.mean(df[df.category == category].text.str.split().str.len()))
    plt.figure(figsize=(15, 5))
    plt.bar(categories, word_averages, color=cm.Paired.colors)
    plt.show()

def print_texts_examples(df: pd.DataFrame):
    categories = np.unique(df.category)
    texts = []
    for category in categories:
        texts.append([category, df[df.category == category][0:1].text.iloc[0]])
    print(tabulate(texts, headers=["Category", "text example"]))

def show_most_frequent_words(df: pd.DataFrame, method: str = 'bag_of_word', nb_features: int = 10, n_gram_range: tuple = (1, 1)):
    if method == 'bag_of_word':
        vectorizer = CountVectorizer(ngram_range=n_gram_range, max_features=nb_features)
        X = vectorizer.fit_transform(df.text)
        df_cv = pd.DataFrame(np.sum(X, axis=0), columns=vectorizer.get_feature_names())
        frequency_cv = df_cv.iloc[:, np.argsort(df_cv.loc[0])]
        print("Bag Of Word: word count with n-grams:", n_gram_range, "\n", frequency_cv.loc[0][::-1], "\n")
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=nb_features)
        X = vectorizer.fit_transform(df.text)
        df_tf = pd.DataFrame(np.sum(X, axis=0), columns=vectorizer.get_feature_names())
        frequency_tf = df_tf.iloc[:, np.argsort(df_tf.loc[0])]
        print("TFIDF: word frequency\n", frequency_tf.loc[0][::-1], "\n")
    else:
        raise Exception('Wrong method')

# ## Print Data Information

texts, categories = load_texts("data/Tobacco3482-OCR/")
df = create_data_frame(texts, categories)
print(df.describe(), "\n")
print_average_number_words(df)
plot_category_histogram(df)
#print_texts_examples(df)
show_most_frequent_words(df, method='bag_of_word', nb_features=20)
show_most_frequent_words(df, method='tfidf', nb_features=20)
show_most_frequent_words(df, method='bag_of_word', nb_features=20, n_gram_range=(2, 2))

# ## Splitting data into a training and testing set 

def hyperparameter_optimization(X_train, y_train):
    # Hyperameters optimization with GridSearchCV = parallel processing
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': [0.1, 0.2],
        'vect__min_df': [0., 0.1],
        'vect__max_features': [1600, 1800, 2000],
        'clf__alpha': [0.10, 0.15, 0.20],
        'tfidf__use_idf': [True]
    }
    if __name__ == "__main__":
        # multiprocessing requires the fork to happen in a __main__ protected
        # block

        # find the best parameters for both the feature extraction and the
        # classifier
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, error_score=0)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Split the dataset, create X (features) and y (target)
X_train, X_test, y_train, y_test = train_test_split(df.text, df.category, test_size=0.2)

#hyperparameter_optimization(X_train, y_train)

# ## Bag of Word representation

# Create document vectors
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=2000, min_df=0, max_df=0.1)
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# train a Naive Bayes classifier
clf = MultinomialNB(alpha=0.15)
clf.fit(X_train_counts, y_train)

y_hat_train_bw = clf.predict(X_train_counts)
error_train_bw = y_hat_train_bw[y_hat_train_bw == y_train].size / y_train.size

y_hat_test_bw = clf.predict(X_test_counts)
error_test_bw = y_hat_test_bw[y_hat_test_bw == y_test].size / y_test.size

table = [["training error", error_train_bw], ["test error", error_test_bw]]
print(tabulate(table, headers=["Bag of Word results"]))

report = classification_report(y_test, y_hat_test_bw)
print(report)
confusion = confusion_matrix(y_test, y_hat_test_bw)
plt.imshow(confusion)
plt.colorbar()
plt.show()

# ## TF-IDF representation

tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_test_tf = tf_transformer.transform(X_test_counts)

# train a Naive Bayes classifier
clf = MultinomialNB(alpha=0.15)
clf.fit(X_train_tf, y_train)

y_hat_train_tf = clf.predict(X_train_tf)
error_train_tf = y_hat_train_tf[y_hat_train_tf == y_train].size / y_train.size

y_hat_test_tf = clf.predict(X_test_tf)
error_test_tf = y_hat_test_tf[y_hat_test_tf == y_test].size / y_test.size

table = [["training error", error_train_tf], ["test error", error_test_tf]]
print(tabulate(table, headers=["TF-IDF results"]))

report = classification_report(y_test, y_hat_test_tf)
print(report)
confusion = confusion_matrix(y_test, y_hat_test_tf)
plt.imshow(confusion)
plt.colorbar()
plt.show()
