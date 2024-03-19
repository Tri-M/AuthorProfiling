import os
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def get_documents(path):
    files = [f for f in os.listdir(documents_path)]
    documents = []
    for f in files[:2]:    #parsing only 10 xml files for fast execution 
        with open(os.path.join(documents_path, f), 'r', encoding='utf-8') as file:
            xml_content = file.read()
            root = ET.fromstring(xml_content)
            document_elements = root.findall('.//document')
            # Extract text content from each <document> element
            document_text=""
            for document_element in document_elements:
                # Decode the CDATA content
                tweet_text = ET.tostring(document_element, method='text', encoding='utf-8').decode('utf-8')
                tweet_text = tweet_text.strip()
                document_text+=tweet_text
            documents.append(document_text)
    # print(document_text)
    # print("///////////////////////////////////////////////////////////////////////////////")
    # txt=ET.tostring(root, encoding='utf-8').decode('utf-8')   #print text content of xml file
    # print(txt)
    return documents

def remove_stop_words(corpus, alpha = 0.01):
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    cleaned_corpus = []
    for document in corpus:
        words = document.split()  # Split the document into words
        cleaned_words = [word for word in words if word.lower() not in stop_words]  # Filter out stop words
        cleaned_document = ' '.join(cleaned_words)  # Join the cleaned words back into a string
        cleaned_corpus.append(cleaned_document)
    return cleaned_corpus


def tokenize(document):
    import re
    document=re.sub('[^a-zA-Z0-9\s]', '', document).lower()
            
    return document

def stem(corpus):
    stemmer = PorterStemmer()
    # Perform stemming on the doc
    stemmed_corpus = []
    for document in corpus:
        words = document.split()  # Split the document into words
        stemmed_words = [stemmer.stem(word) for word in words]  # Stem each word
        stemmed_document = ' '.join(stemmed_words)  # Join the stemmed words back into a string
        stemmed_corpus.append(stemmed_document)
    return stemmed_corpus

def tf_idf(corpus):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the stemmed corpus and transform the corpus into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix,feature_names

def svd(tfidf_matrix):
    # Create an SVD model with the desired number of components
    svd_model = TruncatedSVD(n_components=5)

    # Fit the SVD model to the TF-IDF matrix
    svd_matrix = svd_model.fit_transform(tfidf_matrix)
    return svd_matrix
def lsa(tfidf_matrix):
    svd_model = TruncatedSVD(n_components=5)

    # Fit the SVD model to the TF-IDF matrix
    lsa_matrix = svd_model.fit_transform(tfidf_matrix)
    return lsa_matrix


documents_path = r'D:/sem_8/Information Retrieval/pan18-author-profiling-training-dataset-2018-02-27/pan18-author-profiling-training-dataset-2018-02-27/en/text'
documents=get_documents(documents_path)

corpus = [tokenize(document) for document in documents]
# print("\n\nAfter tokenizing:\n\n",corpus[0])
# print(len(corpus[0]))

corpus = remove_stop_words(corpus)
# print("\n\nAfter stop words removal:\n\n",corpus[0])
# print(len(corpus[0]))

corpus=stem(corpus)
# print("\n\nAfter stemming:\n\n",corpus[0])
# print(len(corpus[0]))

tfidf_matrix,feature_names=tf_idf(corpus)
# Print TF-IDF matrix
print("\nTF-IDF Matrix:\n")
print(tfidf_matrix.toarray())

# Print feature names
# print("\nFeature Names:\n\n",feature_names)

svd_matrix=svd(tfidf_matrix)
# Print the SVD matrix
print("\n\nSVD Matrix:\n",svd_matrix)

lsa_matrix=lsa(tfidf_matrix)

# Print the LSA matrix
print("\n\nLSA Matrix:\n")
print(lsa_matrix)