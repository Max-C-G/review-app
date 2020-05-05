from flask import Flask, render_template
from flask import request
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
import pickle

# load the model from disk
model_file = 'nb_model_final.sav'
vocab_file = 'vocabulary.p'
nb_classifier = pickle.load(open(model_file, 'rb'))
vocab = pickle.load(open(vocab_file, 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_index():
    # print('testing')
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_score():
    # print('review: ', request.form['review'])
    review = request.form['review']
    clf = nb_classifier
    # print(result)
    count_vect_test = CountVectorizer(vocabulary=vocab)
    tester_counts = count_vect_test.fit_transform([review])
    tfidf_transformer = TfidfTransformer()
    tester_tfidf = tfidf_transformer.fit_transform(tester_counts)
    prediction = clf.predict(tester_tfidf)
    print('prediction: ', prediction)
    return render_template('index.html', review = request.form['review'], rating = str(prediction[0]))