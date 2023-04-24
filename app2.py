# Importing required Libraries
import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image

nltk.download('stopwords')
nltk.download('wordnet')
stops = stopwords.words('english')


class WebApp:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.input_resume, self.transform_resume, self.vector_input = None, None, None

    def load_pickle_file(self):
        # Loading the pickle file
        self.tfidf = pickle.load(open('vectorized.pkl', 'rb'))
        self.model = pickle.load(open('model.pkl', 'rb'))

    def user_interface(self):
        self.load_pickle_file()
        # Adding Image
        image = Image.open('resume.jpg')

        st.image(image, width=100)

        # Title and Header
        st.title("Resume Screening")

        # Text input
        self.input_resume = st.text_area("Enter the Resume: ", height=450)

    def output(self):
        self.user_interface()
        if st.button('Predict'):

            self.transform_resume = self.nlp_preprocessing(self.input_resume)

            # 2. Vectorize
            self.vector_input = self.tfidf.transform([self.transform_resume])

            # 3. Predict
            result = self.model.predict(self.vector_input)

            # 4. Display
            if result == 0:
                st.header('Advocate')
            elif result == 1:
                st.header('Blockchain')
            elif result == 2:
                st.header('Business Analyst')
            elif result == 3:
                st.header('Civil Engineer')
            elif result == 4:
                st.header('Data Science')
            elif result == 5:
                st.header('Database')
            elif result == 6:
                st.header('DevOps Engineer')
            elif result == 7:
                st.header('DotNet Developer')
            elif result == 8:
                st.header('Electrical Engineer')
            elif result == 9:
                st.header('HR')
            elif result == 10:
                st.header('Hadoop')
            elif result == 12:
                st.header('Mechanical Engineer')
            elif result == 13:
                st.header('Network Security Engineer')
            elif result == 14:
                st.header('Operation Manager')
            elif result == 15:
                st.header('PMO')
            elif result == 16:
                st.header('Python Developer')
            elif result == 17:
                st.header('Quality Assurance')
            elif result == 18:
                st.header('SAP Developer')
            elif result == 19:
                st.header('Sales')
            elif result == 20:
                st.header('Web Designing')
            else:
                st.header('Fake Resume')

# 1. Preprocessing steps
    def nlp_preprocessing(self, text):
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


if __name__ == '__main__':
    # Creating Object instance
    web_app = WebApp()

    web_app.output()


