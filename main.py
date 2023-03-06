# Importing Required Libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
import pickle

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
stops = stopwords.words('english')

# Loading the dataset
df = pd.read_csv("Data/ResumeDataset.csv")

# Getting an overview on the dataset
# Printing Head and Tail
print("\n First 5 data are: \n", df.head())
print("\n Last 5 data are: \n", df.tail())

# Basic info of the dataset
print("\n Basic information of the dataset are: \n")
df.info()

# Getting shape of the dataset
print("\n Shape of the dataset is: \n", df.shape)

# Data Cleaning
# Null values
print("\n Null Values present in the dataset are: \n", df.isnull().sum())
# Duplicated rows
print("\n Duplicated rows present in the dataset are: ", df.duplicated().sum())
print("We can't remove 796 duplicated rows from our dataset as it will be a huge loss of data.\n")

# Merging few categories into one to reduce complexity
# Value count og category column
print("\n Different Value in Category column are: \n", df['Category'].value_counts())

# We can group two category like testing and automation testing into one category i.e, Quality Assurance
df.replace(df['Category'].replace(['Testing', 'Automation Testing'], 'Quality Assurance', inplace=True))

# Similarly, arts and health & fitness can be merged into HR
df.replace(df['Category'].replace(['Arts', 'Health and fitness'], 'HR', inplace=True))

# Similarly, ETL Developer & Database can be merged into Database
df.replace(df['Category'].replace(['ETL Developer'], 'Database', inplace=True))

# Printing Value count in category column
print("\n Total number of categories in Category column are: ", df['Category'].nunique())


# Applying NLP to the dataset
# To remove the url from the resume column
def remove_url(raw_test):
    raw_test = re.sub(r'http\S+\s*', ' ', raw_test)
    raw_test = re.sub(r'www\S+\s*', ' ', raw_test)
    raw_test = re.sub(r'[a-zA-Z0-9]\S+com\s*', '', raw_test)
    return raw_test


df['Resume'] = df['Resume'].apply(remove_url)


# To convert the non-ascii code into ascii
def encode(text):
    text = text.encode('ascii', 'ignore')  # Encoding
    text = text.decode()  # Decoding
    return text


df['Resume'] = df['Resume'].apply(encode)


# To remove punctuations and blank space from the resume column
def remove_tags(raw_test):
    raw_test = re.sub('\s+', ' ', raw_test)  # To remove Blank Space
    raw_test = re.sub('[^\w\s]', '', raw_test)  # To remove Punctuations
    return raw_test


df['Resume'] = df['Resume'].apply(remove_tags)

# To lower the characters in resume column
df['Resume'] = df['Resume'].apply(lambda x: x.lower())


# Tokenizing the word
def tokenize(text):
    text = word_tokenize(text)
    return text


df['Resume'] = df['Resume'].apply(tokenize)


# Applying stopwords
def remove_stopwords(text):
    text = [word for word in text if word not in stops]
    return text


df['Resume'] = df['Resume'].apply(remove_stopwords)


# Applying Lemmatizer on resume column
def lemmatize_words(text):
    lm = WordNetLemmatizer()
    words = [lm.lemmatize(word, pos='v') for word in text]  # pos='v' means verb removing part of speech
    return ' '.join(words)


df['Resume'] = df['Resume'].apply(lemmatize_words)
print("\n Raw text after applying nlp to the Resume column: \n", df['Resume'][0])

# Splitting the dataset
# Applying Label Encoder on Category column
lb = LabelEncoder()
df['Category'] = lb.fit_transform(df['Category'])

# Applying Tfidf on the resume column
tfidf = TfidfVectorizer(max_features=3000)

# Splitting the dataset into x and y
x = tfidf.fit_transform(df['Resume']).toarray()
y = df['Category']

# Splitting the dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(x, y)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Shape of the train and test data
print("\n Shape of X train: ", x_train.shape)
print("\n Shape of Y train: ", y_train.shape)
print("\n Shape of X test: ", x_test.shape)
print("\n Shape of Y test: ", y_test.shape)

# Applying ML Model
print("\n Applying Multinomial NB model \n")
model_ml = MultinomialNB()
# Fitting the data in Multinomial Naive Bayes
model_ml.fit(x_train, y_train)
# Prediction
y_prediction = model_ml.predict(x_test)  # Test data prediction
y_prediction_train = model_ml.predict(x_train)  # Train data prediction

# Performance of Multinomial NB
# Train accuracy
print("\n Train Accuracy of Multinomial NB model: \n", accuracy_score(y_train, y_prediction_train) * 100)
# Test Accuracy
print("\n Test Accuracy of Multinomial NB model: \n", accuracy_score(y_test, y_prediction) * 100)
# Precision score
precision = precision_score(y_test, y_prediction, average='weighted') * 100
# Recall Score
recall = recall_score(y_test, y_prediction, average='weighted') * 100
# F1 Score
f1 = f1_score(y_test, y_prediction, average='weighted') * 100
print(f"\nPrecision Score\t\tRecall Score\t\tF1 Score")
print(f"{precision}\t{recall}\t{f1}\n")

# Pickling the file
pickle.dump(model_ml, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorized.pkl', 'wb'))
