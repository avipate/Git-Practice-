# Importing required Libraries
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

# Downloading required content

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
stops = stopwords.words('english')

# Loading the dataset
df = pd.read_csv("Data/ResumeDataset.csv")

# Printning head and tail
print("\n First 5 data are: \n", df.head())
print("\n Last 5 data are: \n", df.head())