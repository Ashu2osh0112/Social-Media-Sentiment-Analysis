# -*- coding: utf-8 -*-
"""
## Social Media Sentiment Analysis:
"""

! pip install kaggle

"""###Upload your kaggle.json file"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

"""###Importing Twitter Sentiment dataset"""

!kaggle datasets download -d kazanova/sentiment140

from zipfile import ZipFile
dataset = '/content/sentiment140.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

"""###Importing the Dependencies"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

"""###Data Processing"""

twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')

twitter_data.shape

twitter_data.head()

column_names = ['target','ids','date','flag','user','text']
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1',names=column_names)

twitter_data.shape

twitter_data.head()

twitter_data.isnull().sum()

twitter_data['target'].value_counts()

"""###Convert the target "4" to "1"
"""

twitter_data.replace({'target':{4:1}},inplace=True)

twitter_data['target'].value_counts()

"""0 --> Negative Tweet

1 --> Positive Tweet

Stemming

Stemming is a process of reducing a word to its root word

Example : actor, acting, actress = act
"""

port_stem = PorterStemmer()

def stemming(content):

  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

twitter_data.head()

print(twitter_data['stemmed_content'])

print(twitter_data['target'])

X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

print(X)

print(Y)

"""###Splitting the data to training data and test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

print(Y.shape, Y_train.shape, Y_test.shape)

print(X_train)

print(Y_train)

#Converting textual data to numerical data
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train)

print(X_test)

"""Training the machine Learning model

Logistic Regression
"""

model = LogisticRegression(max_iter = 1000)

model.fit(X_train, Y_train)

"""Model evaluation

Accuracy score
"""

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score on the training data :', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score on the training data :', test_data_accuracy)

"""Model accuracy = 77.8%"""

import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

"""Using the saved model for future predictions"""

loaded_model = pickle.load(open('/content/trained_model.sav', 'rb'))

X_new = X_test[200]
print(Y_test[200])

prediction = model.predict(X_new)
print(prediction)

if (prediction[0] == 0):
  print('Negative Tweet')

else:
  print('Positive Tweet')

X_new = X_test[3]
print(Y_test[3])

prediction = model.predict(X_new)
print(prediction)

if (prediction[0] == 0):
  print('Negative Tweet')

else:
  print('Positive Tweet')
