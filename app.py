from flask import Flask, render_template, request, jsonify
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random

app = Flask(__name__, static_url_path='/static')

# Load your pre-trained model and TF-IDF vectorizer
lr_model = joblib.load('your_model.pkl')
tfidf_vectorizer = joblib.load('your_tfidf_vectorizer.pkl')

# Load the "true" and "fake" datasets
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# Add a label to each dataset
true_data['class'] = 1
fake_data['class'] = 0

# Concatenate the two datasets into a single dataframe
data = pd.concat([true_data, fake_data], axis=0)

# Shuffle the data
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)

# Define the percentage of data for testing
test_data_percentage = 0.2

# Create a random sample for testing
test_data = data.sample(frac=test_data_percentage)
data = data.drop(test_data.index)

def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    random_sample_text = ""
    prediction = None

    if request.method == "POST":
        if "random_sample" in request.form:
            random_sample_text = get_random_sample()

        if "classify_text" in request.form:
            user_text = request.form["user_text"]
            user_text = preprocess_text(user_text)

            user_text_vectorized = tfidf_vectorizer.transform([user_text])
            numerical_prediction = lr_model.predict(user_text_vectorized)[0]
            prediction = "Real" if numerical_prediction == 1 else "Fake"

    return render_template('index.html', prediction=prediction, random_sample_text=random_sample_text)

@app.route("/classify_text", methods=["POST"])
def classify_text():
    if request.method == "POST":
        user_text = request.get_json()["text"]
        user_text = preprocess_text(user_text)

        user_text_vectorized = tfidf_vectorizer.transform([user_text])
        numerical_prediction = lr_model.predict(user_text_vectorized)[0]
        prediction = "Real" if numerical_prediction == 1 else "Fake"

        # Create a response string with newlines
        response = '\n"Result": "{}"\n'.format(prediction)

        return response

@app.route("/result/<prediction>")
def result(prediction):
    return render_template('result.html', prediction=prediction)

@app.route("/get_random_sample")
def get_random_sample():
    random_row = data.sample(1)
    random_sample_text = random_row.iloc[0]['text']
    return random_sample_text

if __name__ == "__main__":
    app.run(debug=True)