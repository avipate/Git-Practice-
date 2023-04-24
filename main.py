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


class ResumeScreening:
    def __init__(self, path="Data/ResumeDataset.csv"):
        # Loading the dataset
        self.df = pd.read_csv(path)
        self.x = None
        self.y = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.y_prediction, self.y_prediction_train = None, None
        self.tfidf = None
        self.model = None

    # Overview of the dataset
    def overview(self):
        # Printing Head and Tail
        print("\n First 5 data are: \n", self.df.head())
        print("\n Last 5 data are: \n", self.df.tail())

        # Basic info of the dataset
        print("\n Basic information of the dataset are: \n")
        self.df.info()

        # Getting shape of the dataset
        print("\n Shape of the dataset is: \n", self.df.shape)

    # Data Cleaning
    def data_cleaning(self):
        # Initialize the overview method
        self.overview()

        # Null values
        print("\n Null Values present in the dataset are: \n", self.df.isnull().sum())
        # Duplicated rows
        print("\n Duplicated rows present in the dataset are: ", self.df.duplicated().sum())
        print("We can't remove 796 duplicated rows from our dataset as it will be a huge loss of data.\n")

    # Category column
    def category(self):
        self.data_cleaning()
        # Merging few categories into one to reduce complexity
        # Value count og category column
        print("\n Different Value in Category column are: \n", self.df['Category'].value_counts())

        # We can group two category like testing and automation testing into one category i.e, Quality Assurance
        self.df.replace(
            self.df['Category'].replace(['Testing', 'Automation Testing'], 'Quality Assurance', inplace=True))

        # Similarly, arts and health & fitness can be merged into HR
        self.df.replace(self.df['Category'].replace(['Arts', 'Health and fitness'], 'HR', inplace=True))

        # Similarly, ETL Developer & Database can be merged into Database
        self.df.replace(self.df['Category'].replace(['ETL Developer'], 'Database', inplace=True))

        # Printing Value count in category column
        print("\n Total number of categories in Category column are: ", self.df['Category'].nunique())

    # Applying NLP on Resume column
    def nlp(self):
        self.category()

        # To remove the url from the resume column
        def applying_nlp(raw_text):
            raw_text = re.sub(r'http\S+\s*', ' ', raw_text)
            raw_text = re.sub(r'www\S+\s*', ' ', raw_text)
            raw_text = re.sub(r'[a-zA-Z0-9]\S+com\s*', '', raw_text)

        # To convert the non-ascii code into ascii
            raw_text = str(raw_text).encode('ascii', 'ignore')  # Encoding
            raw_text = raw_text.decode()  # Decoding

        # To remove punctuations and blank space from the resume column
            raw_text = re.sub('\s+', ' ', raw_text)  # To remove Blank Space
            raw_text = re.sub('[^\w\s]', '', raw_text)  # To remove Punctuations

        # To lower the characters in resume column
            raw_text = str(raw_text).lower()

        # Tokenizing the word
            raw_text = word_tokenize(raw_text)

        # Applying stopwords
            raw_text = [word for word in raw_text if word not in stops]

            # Applying Lemmatizer on resume column
            lm = WordNetLemmatizer()
            words = [lm.lemmatize(word, pos='v') for word in raw_text]  # pos='v' means verb removing part of speech

            return ' '.join(words)

        self.df['Resume'] = self.df['Resume'].apply(applying_nlp)

        print("\n Raw text after applying nlp to the Resume column: \n", self.df['Resume'][0])

    # Splitting the dataset
    def splitting_the_data(self):
        self.nlp()

        # Applying Label Encoder on Category column
        lb = LabelEncoder()
        self.df['Category'] = lb.fit_transform(self.df['Category'])

        # Applying Tfidf on the resume column
        self.tfidf = TfidfVectorizer(max_features=3000)

        # Splitting the dataset into x and y
        self.x = self.tfidf.fit_transform(self.df['Resume']).toarray()
        self.y = self.df['Category']

        # Splitting the dataset into test and train
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        # Shape of the train and test data
        print("\n Shape of X train: ", self.x_train.shape)
        print("\n Shape of Y train: ", self.y_train.shape)
        print("\n Shape of X test: ", self.x_test.shape)
        print("\n Shape of Y test: ", self.y_test.shape)

    # Applying ML Model
    def applying_ml_model(self):
        self.splitting_the_data()
        print("\n Applying Multinomial NB model \n")
        self.model = MultinomialNB()
        # Fitting the data in Multinomial Naive Bayes
        self.model.fit(self.x_train, self.y_train)
        # Prediction
        self.y_prediction = self.model.predict(self.x_test)  # Test data prediction
        self.y_prediction_train = self.model.predict(self.x_train)  # Train data prediction

    # Performance of the model
    def performance_of_the_model(self):
        self.applying_ml_model()
        # Performance of Multinomial NB
        # Train accuracy
        print("\n Train Accuracy of Multinomial NB model: \n", accuracy_score(self.y_train, self.y_prediction_train) * 100)
        # Test Accuracy
        print("\n Test Accuracy of Multinomial NB model: \n", accuracy_score(self.y_test, self.y_prediction) * 100)
        # Precision score
        precision = precision_score(self.y_test, self.y_prediction, average='weighted') * 100
        # Recall Score
        recall = recall_score(self.y_test, self.y_prediction, average='weighted') * 100
        # F1 Score
        f1 = f1_score(self.y_test, self.y_prediction, average='weighted') * 100
        print(f"\nPrecision Score\t\tRecall Score\t\tF1 Score")
        print(f"{precision}\t{recall}\t{f1}\n")

    # Pickling the file
    def pickle_file(self):
        self.performance_of_the_model()
        pickle.dump(self.model, open('model.pkl', 'wb'))
        pickle.dump(self.tfidf, open('vectorized.pkl', 'wb'))


if __name__ == '__main__':
    # Creating object instance
    resume_screening = ResumeScreening()

    resume_screening.pickle_file()
