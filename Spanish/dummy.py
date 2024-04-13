import xml.etree.ElementTree as ET
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

import os
import xml.etree.ElementTree as ET
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Spanish stopwords
stop_words = set(stopwords.words('spanish'))
# Initialize Spanish stemmer
stemmer = SnowballStemmer('spanish')

def preprocess_xml_files(folder_path):
    preprocessed_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            author_data = parse_xml(file_path)
            if author_data:
                preprocessed_data.append(author_data)
    return preprocessed_data

def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        author_lang = root.attrib.get('lang', 'unknown')
        author_documents = []
        for document_elem in root.findall('.//document'):
            tweet = preprocess_text(document_elem.text)
            if tweet:
                author_documents.append(tweet)
        return {'lang': author_lang, 'documents': author_documents}
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def preprocess_text(text):
    # Remove opening and closing document tags
    text = re.sub(r'<document>', '', text)
    text = re.sub(r'</document>', '', text)
    # Remove <![CDATA[ and ]]> from the content
    text = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Remove happy and sad smileys
    text = re.sub(r':\)', '', text)
    text = re.sub(r':\(', '', text)
    # Remove tweets or retweets
    text = re.sub(r'\bRT\b', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and perform stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    return ' '.join(stemmed_tokens).strip()

def load_tweets_from_file(output_file_path):
    tweets_per_user = {}
    current_user = None
    with open(output_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('User'):
                current_user = line.split()[1]
                tweets_per_user[current_user] = []
            elif line.startswith('Tweet'):
                split_line = line.split(': ')
                if len(split_line) > 1:
                    tweet_content = split_line[1].strip()
                    tweets_per_user[current_user].append(tweet_content)
    return tweets_per_user

# Load the preprocessed tweets from the output file
output_file_path = 'spanish_output.txt'
tweets_per_user = load_tweets_from_file(output_file_path)


# Calculate TF-IDF vectors for each user's tweets
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors_per_user = {}
for user, tweets in tweets_per_user.items():
    tfidf_vectors = tfidf_vectorizer.fit_transform(tweets)
    tfidf_vectors_per_user[user] = tfidf_vectors

# Apply Latent Semantic Analysis (LSA) to reduce dimensionality
svd = TruncatedSVD(n_components=5)  # Adjust the number of components as needed
lsa_vectors_per_user = {}
for user, tfidf_vectors in tfidf_vectors_per_user.items():
    lsa_vectors = svd.fit_transform(tfidf_vectors)
    lsa_vectors_per_user[user] = lsa_vectors

# Convert LSA vectors to arrays
X = np.vstack([vector for _, vector in lsa_vectors_per_user.items()])
users = list(lsa_vectors_per_user.values())

# Ensure the number of samples in X and y are the same
y = users

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize XGBoost classifier with GPU support
xgb_classifier = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)

# Train the classifier
xgb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
