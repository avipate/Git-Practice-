# Importing required Libraries
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')
stops = stopwords.words('english')

# Loading the model
model = pickle.load(open("model.pkl", 'rb'))
tfidf = pickle.load(open("vectorized.pkl", 'rb'))

# Creating flask app
app = Flask(__name__)


def nlp_preprocessing(text):
    text = re.sub(r'http\S+\s*', ' ', str(text))
    text = re.sub(r'www\S+\s*', ' ', str(text))
    text = re.sub(r'[a-zA-Z0-9]\S+com\s*', '', str(text))

    # To convert the non-ascii code into ascii
    text = text.encode('ascii', 'ignore')  # Encoding
    text = text.decode()  # Decoding

    # To remove punctuations and blank space from the resume column
    text = re.sub('\s+', ' ', str(text))  # To remove Blank Space
    text = re.sub('[^\w\s]', '', str(text))  # To remove Punctuations

    # Tokenizing the word
    text = word_tokenize(text)

    # Applying stopwords
    text = [word for word in text if word not in stops]

    # Applying Lemmatizer on resume column
    lm = WordNetLemmatizer()
    words = [lm.lemmatize(word, pos='v') for word in text]  # pos='v' means verb removing part of speech
    return ' '.join(words)


# Creating home route
@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def prediction_page():
    resume = request.form.get('resume')
    resume = nlp_preprocessing(resume)
    tfidf_vector = tfidf.transform([resume])
    result = model.predict(tfidf_vector)

    return render_template('prediction.html', result="The Person belong to {} domain".format(result))


if __name__ == "__main__":
    app.run(port=3000, debug=True)
