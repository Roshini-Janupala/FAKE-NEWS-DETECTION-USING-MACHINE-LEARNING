import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import string
import joblib

true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')
true_data['class'] = 1
fake_data['class'] = 0
# Display the number of articles in the datasets
print("Number of articles in the True dataset:", len(true_data))
print("Number of articles in the Fake dataset:", len(fake_data))

data = pd.concat([true_data, fake_data], axis=0)
print("Total number of articles in the combined dataset:", len(data))
data.drop(['title', 'subject', 'date'], axis=1, inplace=True)
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

# Create and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()  # Define and create the TF-IDF vectorizer
x = tfidf_vectorizer.fit_transform(x)  # Fit it to your text data

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

'''# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import LogisticRegression

# Train your machine learning model (Logistic Regression)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predict using the trained model
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_pred_lr, y_test)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred_lr)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot()
plt.show()

classification_rep = classification_report(y_test, y_pred_lr)
print(f"Classification Report:\n{classification_rep}")
from sklearn.ensemble import RandomForestClassifier
rfc_model= RandomForestClassifier(n_estimators=100,criterion='entropy')
rfc_model.fit(X_train, y_train)
y_pred_rfc=rfc_model.predict(X_test)
y_pred_rfc
accuracy_score(y_test,y_pred_rfc)
cm = confusion_matrix(y_test, y_pred_rfc)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False, True])

cm_display.plot()
plt.show()
print(classification_report(y_pred_rfc,y_test))'''
#DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train, y_train)
y_pred_dtc=dtc_model.predict(X_test)
y_pred_dtc
accuracy_score(y_pred_dtc,y_test)
cm = confusion_matrix(y_test, y_pred_dtc)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])

cm_display.plot()
plt.show()
classification_rep = classification_report(y_test, y_pred_dtc)
print(f"Classification Report:\n{classification_rep}")
# Save the trained model and vectorizer to .pkl files
joblib.dump(dtc_model, 'your_model.pkl')
joblib.dump(tfidf_vectorizer, 'your_tfidf_vectorizer.pkl')
print("Model and vectorizer are saved.")
print("Number of articles in the training set:", len(y_train))
print("Number of articles in the testing set:", len(y_test))





