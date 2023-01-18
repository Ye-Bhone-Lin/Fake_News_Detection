import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Dense, LSTM,Bidirectional, Dropout
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')
fake_news['classification'] = 0
true_news['classification'] = 1
fake_news.drop_duplicates(inplace=True)
true_news.drop_duplicates(inplace=True)
fake_news = fake_news.drop(labels=['title','date'],axis=1)
true_news = true_news.drop(labels=['title','date'],axis=1)
df = pd.concat([true_news,fake_news],axis=0)
df = df.sample(frac=1)
df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)
st.title('Fake News Detection')
def clean_text(text):
  text = re.sub('\[.*?\]', '', text)
  text = re.sub("\\W"," ",text) 
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text) 
  text = re.sub('[^A-Za-z]+',' ',text)
  text = text.lower()
  return text
df['clean_text'] = df['text'].apply(clean_text)

x = df['clean_text']
y = df['classification']
x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.25,random_state=0)
vectorization = TfidfVectorizer()
x_train_cv = vectorization.fit_transform(x_train)
x_test_cv = vectorization.transform(x_test)
clf = RandomForestClassifier(random_state=0)
clf.fit(x_train_cv, y_train)
predict_y = clf.predict(x_test_cv) 
with st.form('Classification'):
    test_msg = st.text_input('Classification')
    submitted = st.form_submit_button("Submit")
    if submitted:
        text = clean_text(test_msg)
        text = str(text)
        st.write(test_msg)
        st.write(clf.predict(vectorization.transform([text])))

STOPWORDS = set(stopwords.words('english'))
def remove_stopword(text):
  text = ' '.join(word for word in text.split() if word not in STOPWORDS)
  return text
df['stopwords'] = df['clean_text'].apply(remove_stopword)
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True,char_level=False,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(df['stopwords'])
new_model = tf.keras.models.load_model('multi_model')
def final(text):
  list_1 = []
  list_1.append(text)
  seq = tokenizer.texts_to_sequences(list_1)
  padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
  return padded
with st.form('Multi Classification'):

    input1 = final(st.text_input('Multiclassification'))
    submitted = st.form_submit_button("Submit")
    if submitted:
        prediction = new_model.predict(input1)
        labels = ['News', 'politics', 'Government News', 'left-news', 'US_News','Middle-east','wordlnews','politicsNews']
        st.write((labels[np.argmax(prediction)]))
