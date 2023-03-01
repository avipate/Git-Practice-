# Importing required Libraries
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')
stops = stopwords.words('english')

# Loading the model
model = pickle.load(open("model.pkl", 'rb'))


# def remove_url(raw_test):
#     raw_test = re.sub(r'http\S+\s*', ' ', raw_test)
#     raw_test = re.sub(r'www\S+\s*', ' ', raw_test)
#     raw_test = re.sub(r'[a-zA-Z0-9]\S+com\s*', '', raw_test)
#     return raw_test


# To convert the non-ascii code into ascii
def encode(text):
    text = text.encode('ascii', 'ignore')  # Encoding
    text = text.decode()  # Decoding
    return text


# To remove punctuations and blank space from the resume column
def remove_tags(raw_test):
    raw_test = re.sub('\s+', ' ', raw_test)  # To remove Blank Space
    raw_test = re.sub('[^\w\s]', '', raw_test)  # To remove Punctuations
    return raw_test


# To lower the characters in resume column
# def to_lower(text):
#
#     return text
# df['Resume'] = df['Resume'].apply(lambda x: x.lower())


# Tokenizing the word
def tokenize(text):
    text = word_tokenize(text)
    return text


# Applying stopwords
def remove_stopwords(text):
    text = [word for word in text if word not in stops]
    return text


# Applying Lemmatizer on resume column
def lemmatize_words(text):
    lm = WordNetLemmatizer()
    words = [lm.lemmatize(word, pos='v') for word in text]  # pos='v' means verb removing part of speech
    return ' '.join(words)


# resume = request.form.get('resume')
# print(resume)
# Creating app
app = Flask(__name__)


# Creating home route
@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def prediction_page():
    resume = request.form.get('resume')
    # resume = remove_url(resume)

    def remove_url(raw_test):
        raw_test = re.sub(r'http\S+\s*', ' ', raw_test)
        raw_test = re.sub(r'www\S+\s*', ' ', raw_test)
        raw_test = re.sub(r'[a-zA-Z0-9]\S+com\s*', '', raw_test)
        return raw_test
    resume = remove_url(resume)
    resume = encode(resume)
    resume = remove_tags(resume)
    # resume = str(lambda x: x.lower())
    resume = tokenize(resume)
    resume = remove_stopwords(resume)
    resume = lemmatize_words(resume)
    print(resume)

    return render_template('prediction.html')


if __name__ == "__main__":
    app.run(port=3000, debug=True)
