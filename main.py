import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset
# data = pd.read_csv('dataset/spam.csv', sep=',', encoding='utf-8')
data = pd.read_csv('dataset/spam.csv', sep=',', encoding='latin1')

# Rename the columns
# data.columns = ['label', 'message']
data.columns = ['label', 'message', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']

# Keeping necessary columns only
data = data[['label', 'message']]

# Check the first few rows to understand the structure
print(data.head())

# Convert labels to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split features and labels
X = data['message']
y = data['label']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the text data
X_tfidf = tfidf.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics for Logistic Regression
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print performance metrics for Logistic Regression
print('\n')
print(f'Logistic Regression Metrics:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Initialize Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_nb = nb_classifier.predict(X_test)

# Calculate evaluation metrics for Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

# Print evaluation metrics for Naive Bayes
print('\n')
print(f'Naive Bayes Classifier Metrics:')
print(f'  Accuracy: {accuracy_nb:.4f}')
print(f'  Precision: {precision_nb:.4f}')
print(f'  Recall: {recall_nb:.4f}')
print(f'  F1 Score: {f1_nb:.4f}')

# Save the trained model and the TF-IDF vectorizer
joblib.dump(model, 'models/logistic_regression_spam_classifier.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
