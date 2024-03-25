import xml.etree.ElementTree as ET
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

import os
import xml.etree.ElementTree as ET
import re

nltk.download('stopwords')
nltk.download('punkt')

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
    # Remove slangs
    
    # slangs = {
    #     'lol': '',
    #     'rofl': '',
    #     'omg': '',
    #     'brb': '',
    # }
    # for slang, replacement in slangs.items():
    #     text = re.sub(r'\b{}\b'.format(slang), replacement, text, flags=re.IGNORECASE)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and perform stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    return ' '.join(stemmed_tokens).strip()


folder_path = 'F:/pan18-author-profiling-training-dataset-2018-02-27/pan18-author-profiling-training-dataset-2018-02-27/es/text'
output_file_path = 'F:/AP/AuthorProfiling/Spanish/spanish_output.txt'

preprocessed_data = preprocess_xml_files(folder_path)

with open(output_file_path, 'w', encoding='utf-8') as f:
    for idx, author_data in enumerate(preprocessed_data):
        f.write(f"User {idx + 1}: {author_data['lang']}\n")
        for i, tweet in enumerate(author_data['documents'], start=1):
            f.write(f"Tweet {i}: {tweet}\n\n")

print("Output written to", output_file_path)

#________________________________________________________________
#TF-IDF


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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


# Function to calculate TF-IDF vectors for each user
def calculate_tfidf_vectors(tweets_per_user):
    tfidf_vectors_per_user = {}
    tfidf_vectorizer = TfidfVectorizer()
    for user, tweets in tweets_per_user.items():
        # Fit TF-IDF vectorizer on the user's tweets
        tfidf_vectors = tfidf_vectorizer.fit_transform(tweets)
        # Store the TF-IDF vectors
        tfidf_vectors_per_user[user] = (tfidf_vectorizer, tfidf_vectors)
    return tfidf_vectors_per_user


# Function to print words along with their TF-IDF scores
def print_words_with_tfidf(tfidf_vectors_per_user, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        for user, (tfidf_vectorizer, tfidf_vectors) in tfidf_vectors_per_user.items():
            f.write(f"User {user}:\n")
            feature_names = tfidf_vectorizer.get_feature_names_out()
            vocabulary = tfidf_vectorizer.vocabulary_
            for i, vector_row in enumerate(tfidf_vectors):
                f.write(f"Tweet {i + 1}:\n")
                
                indices = vector_row.indices
                tfidf_scores = vector_row.data
                sorted_indices = np.argsort(tfidf_scores)[::-1]
                for idx in sorted_indices:
                    feature_index = indices[idx]
                    if feature_index in vocabulary:
                        word = feature_names[vocabulary[feature_index]]
                        score = tfidf_scores[idx]
                        f.write(f"{word}: {score}\n")
            f.write("\n")


tweets_per_user = load_tweets_from_file(output_file_path)

tfidf_vectors_per_user = calculate_tfidf_vectors(tweets_per_user)

output_tfidf_file_path = 'F:/AP/AuthorProfiling/Spanish/spanish_tfidf_output.txt'
print_words_with_tfidf(tfidf_vectors_per_user, output_tfidf_file_path)

print("TF-IDF words and scores written to", output_tfidf_file_path)

# Function to print the TF-IDF matrix with words
# Function to print the TF-IDF matrix with words
def print_tfidf_matrix_with_words(tfidf_vectors_per_user, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for user, (tfidf_vectorizer, tfidf_vectors) in tfidf_vectors_per_user.items():
            f.write(f"User {user}:\n")
            feature_names = tfidf_vectorizer.get_feature_names_out()
            for i, vector_row in enumerate(tfidf_vectors):
                f.write(f"Tweet {i + 1}: ")
                words_with_scores = [(feature_names[idx], score) for idx, score in zip(vector_row.indices, vector_row.data) if idx < len(feature_names)]
                words_with_scores.sort(key=lambda x: x[0])  # Sort by word
                for word, tfidf_score in words_with_scores:
                    f.write(f"{word}({tfidf_score:.2f}) ")
                f.write("\n")
            f.write("\n")


output_tfidf_matrix_with_words_file_path = 'F:/AP/AuthorProfiling/Spanish/spanish_tfidf_matrix_with_words.txt'
print_tfidf_matrix_with_words(tfidf_vectors_per_user, output_tfidf_matrix_with_words_file_path)

print("TF-IDF matrix with words written to", output_tfidf_matrix_with_words_file_path)


#__________________________________________

from sklearn.decomposition import TruncatedSVD

# Function to perform SVD for each user
def perform_svd(tfidf_vectors_per_user, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for user, (tfidf_vectorizer, tfidf_vectors) in tfidf_vectors_per_user.items():
            f.write(f"User {user}:\n")
            svd = TruncatedSVD(n_components=5)  
            svd_vectors = svd.fit_transform(tfidf_vectors)
            for i, svd_vector in enumerate(svd_vectors):
                f.write(f"SVD Component {i + 1}:\n")
                for j, value in enumerate(svd_vector):
                    f.write(f"Value {j + 1}: {value}\n")
            f.write("\n")


# Perform SVD for each user's TF-IDF vectors
output_svd_file_path = 'F:/AP/AuthorProfiling/Spanish/spanish_svd.txt'
perform_svd(tfidf_vectors_per_user, output_svd_file_path)
print("SVD components written to", output_svd_file_path)

from sklearn.decomposition import TruncatedSVD

def perform_lsa_with_svd(tfidf_vectors_per_user, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for user, (tfidf_vectorizer, tfidf_vectors) in tfidf_vectors_per_user.items():
            f.write(f"User {user}:\n")
            svd = TruncatedSVD(n_components=5)  
            svd_vectors = svd.fit_transform(tfidf_vectors)
            
            lsa = TruncatedSVD(n_components=5)  
            lsa_vectors = lsa.fit_transform(svd_vectors)
            
            for i, lsa_vector in enumerate(lsa_vectors):
                f.write(f"LSA Component {i + 1}:\n")
                for j, value in enumerate(lsa_vector):
                    f.write(f"Value {j + 1}: {value}\n")
            f.write("\n")



output_lsa_file_path = 'F:/AP/AuthorProfiling/Spanish/spanish_lsa.txt'
perform_lsa_with_svd(tfidf_vectors_per_user, output_lsa_file_path)
print("LSA components written to", output_lsa_file_path)

#Support Vector Machine Classification

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def parse_lsa_file(filename):
    lsa_components_dict = {}
    current_user_label = None
    current_lsa_values = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("User"):
                if current_user_label is not None:
                    lsa_components_dict[current_user_label] = current_lsa_values
                    current_lsa_values = []  
                current_user_label = int(line.split()[1].replace("::", ""))  
            elif line.startswith("Value"):
                _, lsa_value = line.split(": ")
                current_lsa_values.append(float(lsa_value))

        if current_user_label is not None:
            lsa_components_dict[current_user_label] = current_lsa_values

    return lsa_components_dict

filename = "Spanish/spanish_lsa.txt"
lsa_components_dict = parse_lsa_file(filename)

# print(lsa_components_dict)
# print("Number of users with LSA components:", len(lsa_components_dict))
# print("User labels:", list(lsa_components_dict.keys()))
# print("Number of components for user 1:", len(lsa_components_dict[1]))
# print("LSA components for user 1:", lsa_components_dict[1])


def read_gender_file(file_path):
    gender_labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            user_id, gender = line.strip().split(':::')
            gender_labels[user_id] = gender
    return gender_labels



gender_labels = read_gender_file('F:\pan18-author-profiling-training-dataset-2018-02-27\pan18-author-profiling-training-dataset-2018-02-27\es\es.txt')

X = []  # Features
y = []  # Labels


for user_label, values in lsa_components_dict.items():
    for value_list in values:
        X.append(value_list)
        y.append(user_label)


X = []  # Features
y = []  # Labels


for i, (user_label, values) in enumerate(lsa_components_dict.items()):
    for value_list in values:
        X.append(value_list)
        y.append(user_label)
        print(f"Iteration: {i+1}/{len(lsa_components_dict)}")



X = np.array(X)
y = np.array(y)

# Reshape X to have two dimensions
X = X.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

from sklearn.ensemble import RandomForestClassifier
import joblib

# Fitting the Random Forest classifier
print("Fitting Random Forest classifier...")  
rf_classifier = RandomForestClassifier(n_estimators=1, random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest classifier fitting completed.")  

# Save the trained Random Forest model
rf_model_file = 'rf_model.pkl'
joblib.dump(rf_classifier, rf_model_file)
print("Random Forest model saved to", rf_model_file)

# Load the saved Random Forest model
loaded_rf_model = joblib.load(rf_model_file)
print("Random Forest model loaded successfully.")

# Predict using the loaded model
y_pred = loaded_rf_model.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
