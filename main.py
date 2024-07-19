import streamlit as st
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Download NLTK stopwords
nltk.download('stopwords')

# Load data
data = pd.read_csv("Data_Set.csv")
data["labels"] = data["class"].map({0: "Hate Speech Detected", 1: "Offensive Language", 2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

# Define offensive words for the profanity filter
offensive_words = ["kill", "badword1", "badword2", "badword3"]  # Add your list of offensive words here

# Profanity filter function
def profanity_filter(text):
    pattern = re.compile("|".join(offensive_words), re.IGNORECASE)
    filtered_text = pattern.sub(lambda x: '*' * len(x.group()), text)
    return filtered_text

# Text cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply text cleaning
data["tweet"] = data["tweet"].apply(clean)

# Prepare data for training
x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Streamlit app for hate speech detection
def hate_speech_detection():
    st.title("AI Based Offensive Word and Hate Speech Management System on Online Platforms Using Profanity Filter Algorithm")
    user = st.text_area("Enter a post: ")
    if len(user) < 1:
        st.write(" ")
    else:
        # Apply profanity filter to the user input
        filtered_user = profanity_filter(user)
        
        # Display the filtered input with censored profanity
        st.subheader("Filtered Input:")
        st.write(filtered_user)
        
        # Clean the original input text (not the filtered one for classification)
        sample = clean(user)
        
        # Transform the cleaned text to match the training data format
        data = cv.transform([sample]).toarray()
        
        # Predict the class
        prediction = clf.predict(data)
        
        st.subheader("Prediction:")
        st.write(prediction[0])

# Run the Streamlit app
hate_speech_detection()
