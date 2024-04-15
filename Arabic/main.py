import xml.etree.ElementTree as ET
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import os
import xml.etree.ElementTree as ET
import re

nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words('arabic'))
# Initialize english stemmer
stemmer = ISRIStemmer()

def preprocess_xml_files(folder_path):
    preprocessed_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            author_data = parse_xml(file_path)
            if author_data:
                preprocessed_data[filename[:-4]]=author_data
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


folder_path = r'F:\pan18-author-profiling-training-dataset-2018-02-27\pan18-author-profiling-training-dataset-2018-02-27\ar\text'
output_file_path = 'mf_arabic_output.txt'

preprocessed_data = preprocess_xml_files(folder_path)

with open(output_file_path, 'w', encoding='utf-8') as f:
    for user, author_data in preprocessed_data.items():
        f.write(f"User {user}: {author_data['lang']}\n")
        for i, tweet in enumerate(author_data['documents'], start=1):
            f.write(f"Tweet {i}: {tweet}\n\n")

print("Output written to", output_file_path)

#________________________________________________________________
# TF-IDF


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
                # Sort indices by TF-IDF value in descending order
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

# Load the preprocessed tweets from the output file
tweets_per_user = load_tweets_from_file(output_file_path)

# Calculate TF-IDF vectors for each user's tweets
tfidf_vectors_per_user = calculate_tfidf_vectors(tweets_per_user)

# Write the results to a new file
output_tfidf_file_path = 'mf_arabic_tfidf_output.txt'
print_words_with_tfidf(tfidf_vectors_per_user, output_tfidf_file_path)

print("TF-IDF words and scores written to", output_tfidf_file_path)

# #__________________________________________

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
output_svd_file_path = 'mf_arabic_svd.txt'
perform_svd(tfidf_vectors_per_user, output_svd_file_path)
print("SVD components written to", output_svd_file_path)

from sklearn.decomposition import TruncatedSVD

# Function to perform LSA for each user
def perform_lsa(tfidf_vectors_per_user, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for user, (tfidf_vectorizer, tfidf_vectors) in tfidf_vectors_per_user.items():
            f.write(f"User {user}:\n")
            
            combined_tfidf_vectors = tfidf_vectors[:100]  
            svd = TruncatedSVD(n_components=5)  
            lsa_vectors = svd.fit_transform(combined_tfidf_vectors)
            for i, lsa_vector in enumerate(lsa_vectors):
                f.write(f"LSA Component {i + 1}:\n")
                for j, value in enumerate(lsa_vector):
                    f.write(f"Value {j + 1}: {value}\n")
            f.write("\n")

# Perform LSA for each user's TF-IDF vectors
output_lsa_file_path = 'mf_arabic_lsa.txt'
perform_lsa(tfidf_vectors_per_user, output_lsa_file_path)
print("LSA components written to", output_lsa_file_path)
# Define the filename
filename = "F:\AP\AuthorProfiling\Arabic\mf_arabic_lsa.txt"
# Initialize an empty dictionary to store LSA components
lsa_components_dict = {}

# Open the text file and read its contents
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse the lines to extract user IDs and LSA components
user_id = None
lsa_components = []
values=[]
for line in lines:
    if len(values)==5:
        # print("lsa comp")
        lsa_components.append(values)
        values=[]
    if line.startswith('User'):
        # Extract user ID from the line
        user_id = line.split('User ')[1].split('::')[0]
        # print("user id")
    elif line.startswith('LSA Component'):
        # Skip the line containing 'LSA Component'
        pass
    elif line.startswith('Value'):
        # Extract LSA component values
        # print("value")
        value = line.split(': ')[1].strip()
        # print("Values : ",value)
        values.append(float(value))
        # End of user data, add to dictionary
    elif line.startswith(''):
        if lsa_components!=[]:
            lsa_components_dict[user_id] = lsa_components
            lsa_components = []

# Print the dictionary
# print(lsa_components_dict)
 
# # print(len(lsa_components_dict))
# # print(len(lsa_components_dict.values()))
# # print(len(lsa_components_dict[1]))
# # print(len(lsa_components_dict[3000]))
# # print(lsa_components_dict[1])

# from sklearn import svm
# import numpy as np

# # Assuming your data is stored in a dictionary called 'data_dict'
# # where keys are user labels and values are lists of 100 elements
# # where each element is a list of 5 float values.

# Step 1: Prepare the Data
X = []  # Features
y = []  # Labels

for user_label, values in lsa_components_dict.items():
    if len(values)<100:
            values.append([0,0,0,0,0])
            # print("Culprit",user_label,len(values))
    for value_list in values:
        if len(value_list)==5:
            X.append(value_list)
            y.append(user_label)
            

# # Pad elements with length less than 100 with zeros
print("len :",len(X))

# Read gender information from the text file and map 'male' and 'female' to 'm' and 'f'
gender_dict = {}
with open("F:\pan18-author-profiling-training-dataset-2018-02-27\pan18-author-profiling-training-dataset-2018-02-27\ar\ar.txt", 'r') as gender_file:
    for line in gender_file:
        user_id, gender = line.strip().split(':::')
        gender_dict[user_id] = gender.strip()[0].lower()  # Extract the first character and convert to lowercase

# Create a list of corresponding genders ('m' or 'f')
y = [gender_dict.get(user_id, 'U') for user_id in y]


# Convert X to numpy array
import numpy as np
X = np.array(X)
y = np.array(y)
# print(X)
print("\n\n________________________________________________________")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
# Create a Random Forest classifier with adjusted parameters
rf_classifier = RandomForestClassifier(n_estimators=200, 
                                        max_depth=20, 
                                        min_samples_split=5, 
                                        max_features="sqrt", 
                                        max_samples=0.8, 
                                        random_state=42, 
                                        verbose=1)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = rf_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

import pickle

with open(r'D:\sem_8\Information Retrieval\package\AuthorProfiling\Arabic\rf_model_200xDT_mDepth20_sam8_29_03.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)