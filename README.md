# Hate Speech Detection

This project implements a hate speech detection model using a decision tree classifier and Twitter data.

## Preview

![image](images/Screenshot%20(98).png)

![image](images/Screenshot%20(97).png)

![image](images/Screenshot%20(96).png)

![image](images/Screenshot%20(93).png)

## Project Overview

The code consists of the following major components:

1. Reading and Preprocessing the Data:
   - The code reads the Twitter data from a CSV file (`Data_Set.csv`).
   - It maps the class labels (0, 1, 2) to meaningful categories: Hate Speech, Offensive Language, and No Hate and Offensive.
   - The tweets and corresponding labels are extracted and stored in a pandas DataFrame.

2. Text Preprocessing:
   - The text preprocessing function (`clean`) is defined to clean and transform the tweets.
   - The steps include converting text to lowercase, removing URLs, HTML tags, punctuation, and numbers.
   - Stop words from the NLTK corpus are removed, and words are stemmed using the SnowballStemmer.

3. Feature Extraction:
   - The CountVectorizer from scikit-learn is used to transform the preprocessed text into numerical features.
   - The feature matrix (`X`) is created by fitting the vectorizer on the preprocessed tweet text.

4. Data Splitting:
   - The dataset is split into training and testing sets using the train_test_split function.
   - The feature matrix (`X`) and the corresponding labels (`y`) are divided into X_train, X_test, y_train, and y_test.

5. Model Training and Evaluation:
   - A DecisionTreeClassifier is initialized and trained on the training set.
   - The model's accuracy is calculated by scoring it on the testing set.

6. Hate Speech Detection Web Application:
   - A Streamlit web application is created to interactively detect hate speech in user-provided tweets.
   - Users can enter any input in the provided text area.
   - The application uses the trained model and the CountVectorizer to preprocess and classify the user's input.
   - The predicted label for the input is displayed on the screen.

## Usage

To use this code, follow these steps:

1. Install the required packages by running the following command:
```shell
   pip install -r requirements.txt
```

2. Download the dataset file `twitter.csv` and place it in the same directory as the code file.

3. Run the code using the following command:
```shell
   streamlit run main.py
```

4. The Streamlit web application will open in your browser.

5. Enter an input in the provided text area to detect whether it contains hate speech.

6. The model will predict the label for the tweet, and the predicted category will be displayed on the screen.

