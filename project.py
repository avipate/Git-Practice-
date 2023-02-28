# Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import SparseCategoricalCrossentropy
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
stops = stopwords.words('english')


# Creating class for dataset
class Resume:
    def __init__(self, filepath="Data/ResumeDataset.csv"):
        # Loading the dataset
        self.df = pd.read_csv(filepath)
        # creating the deep learning model variable
        self.model_dl = Sequential()
        # Creating the machine learning model variable
        self.model_ml = MultinomialNB()
        # Creating other required Variables
        self.category = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.history = None
        self.y_prediction = None
        self.y_prediction_train = None

    # getting an overview on the dataset
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
        # Null values
        print("\n Null Values present in the dataset are: \n", self.df.isnull().sum())
        # Duplicated rows
        print("\n Duplicated rows present in the dataset are: ", self.df.duplicated().sum())
        print("We can't remove 796 duplicated rows from our dataset as it will be a huge loss of data.\n")

    # Data visualization
    def visualization(self):
        # Printing value counts in the category column
        print("\n Value count in the Category column are: \n", self.df['Category'].value_counts())

        # Visualization of category column
        plt.figure(figsize=(8, 12))
        self.category = self.df['Category'].value_counts().plot(kind='pie', autopct='%.2f')
        plt.title('Different types of Category Distribution')
        plt.show()

    # Category
    def category_rename(self):
        # We can group two category like testing and automation testing into one category i.e, Quality Assurance
        self.df.replace(self.df['Category'].replace(['Testing', 'Automation Testing'],
                                                    'Quality Assurance', inplace=True))
        # Similarly, arts and health & fitness can be merged into HR
        self.df.replace(self.df['Category'].replace(['Arts', 'Health and fitness'], 'HR', inplace=True))
        # Printing Value count in category column
        print("\n Total number of categories in Category column are: ", self.df['Category'].nunique())
        # Data Visualization after merging
        sns.countplot(data=self.df, y=self.df['Category'])
        plt.title("Distribution of Category")
        plt.grid()
        plt.show()

    # Applying NLP
    def nlp(self):
        # To remove the url from the resume column
        def remove_url(raw_test):
            raw_test = re.sub(r'http\S+\s*', ' ', raw_test)
            raw_test = re.sub(r'www\S+\s*', ' ', raw_test)
            raw_test = re.sub(r'[a-zA-Z0-9]\S+com\s*', '', raw_test)
            return raw_test

        self.df['Resume'] = self.df['Resume'].apply(remove_url)

        # To convert the non-ascii code into ascii
        def encode(text):
            text = text.encode('ascii', 'ignore')  # Encoding
            text = text.decode()  # Decoding
            return text

        self.df['Resume'] = self.df['Resume'].apply(encode)

        # To remove punctuations and blank space from the resume column
        def remove_tags(raw_test):
            raw_test = re.sub('\s+', ' ', raw_test)  # To remove Blank Space
            raw_test = re.sub('[^\w\s]', '', raw_test)  # To remove Punctuations
            return raw_test

        self.df['Resume'] = self.df['Resume'].apply(remove_tags)

        # To lower the characters in resume column
        self.df['Resume'] = self.df['Resume'].apply(lambda x: x.lower())

        # Tokenizing the word
        def tokenize(text):
            text = word_tokenize(text)
            return text

        self.df['Resume'] = self.df['Resume'].apply(tokenize)

        # Applying stopwords
        def remove_stopwords(text):
            text = [word for word in text if word not in stops]
            return text

        self.df['Resume'] = self.df['Resume'].apply(remove_stopwords)

        # Applying Lemmatizer on resume column
        def lemmatize_words(text):
            lm = WordNetLemmatizer()
            words = [lm.lemmatize(word, pos='v') for word in text]  # pos='v' means verb removing part of speech
            return ' '.join(words)

        self.df['Resume'] = self.df['Resume'].apply(lemmatize_words)
        print("\n Raw text after applying nlp to the Resume column: \n", self.df['Resume'][0])

    # splitting the dataset
    def split(self):
        # Applying Label Encoder on Category column
        lb = LabelEncoder()
        self.df['Category'] = lb.fit_transform(self.df['Category'])
        # Applying Tfidf on the resume column
        tfidf = TfidfVectorizer()
        # Splitting the dataset into x and y
        self.x = tfidf.fit_transform(self.df['Resume']).toarray()
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

    # Applying Machine Learning model
    def applying_ml(self):
        print("-------------------------------------------------------------------")
        print("\n Applying Multinomial NB model \n")
        # Fitting the data in Multinomial Naive Bayes
        self.model_ml.fit(self.x_train, self.y_train)
        # Prediction
        self.y_prediction = self.model_ml.predict(self.x_test)  # Test data prediction
        self.y_prediction_train = self.model_ml.predict(self.x_train)  # Train data prediction

    # Performance of ml model
    def performance_ml(self):
        # Performance of Multinomial NB
        # Train accuracy
        print("\n Train Accuracy of Multinomial NB model: \n",
              accuracy_score(self.y_train, self.y_prediction_train) * 100)
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
        print("-------------------------------------------------------------------")

    # Implementing Deep Learning model
    def applying_dl(self):
        print("\n Applying Deep Learning model \n")
        # Adding Layers to the model
        # Input Layer
        self.model_dl.add(Flatten())
        # Hidden Layer
        self.model_dl.add(Dense(32, activation='relu'))
        self.model_dl.add(Dense(32, activation='relu'))
        # Output Layer
        self.model_dl.add(Dense(22, activation='softmax'))
        # Compiling model
        self.model_dl.compile(loss=SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
        # Fitting the data into the model
        self.history = self.model_dl.fit(self.x_train, self.y_train, epochs=15, batch_size=128,
                                         validation_data=[self.x_test, self.y_test])

    # Performance of dl model
    def performance_dl(self):
        # Model Summary
        print("\n Model Summary: \n")
        self.model_dl.summary()

        # Validated Accuracy and loss of dl model
        val_loss, val_acc = self.model_dl.evaluate(self.x_test, self.y_test, verbose=2)
        print("\n Validated Accuracy of the DL model: \n", val_acc * 100)
        print("\n Validated loss of the DL model: \n ", val_loss)

        # Visualizing accuracy and loss of dl model
        # Accuracy of dl model
        plt.figure(figsize=(8, 8))
        plt.plot(self.history.history['accuracy'], linestyle='dashed')
        plt.plot(self.history.history['val_accuracy'])
        plt.title('DL Model Accuracy')
        plt.ylabel('Accuracy  ---->')
        plt.xlabel('Epoch  ---->')
        plt.grid()
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Loss of dl model
        plt.figure(figsize=(8, 8))
        plt.plot(self.history.history['loss'], linestyle='dashed')
        plt.plot(self.history.history['val_loss'])
        plt.title('DL Model Loss')
        plt.ylabel('Loss  ---->')
        plt.xlabel('Epoch  ---->')
        plt.grid()
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


# Driver Code
if __name__ == "__main__":
    # Creating model instance
    model_instance = Resume()

    # Getting an overview on the dataset
    model_instance.overview()

    # Data Cleaning
    model_instance.data_cleaning()

    # Data Visualization
    model_instance.visualization()

    # Category renaming
    model_instance.category_rename()

    # Applying NLP
    model_instance.nlp()

    # Splitting the dataset
    model_instance.split()

    # Applying ml
    model_instance.applying_ml()

    # Performance of ml model
    model_instance.performance_ml()

    # Applying dl
    model_instance.applying_dl()

    # Performance of dl model
    model_instance.performance_dl()
