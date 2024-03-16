import os
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def get_documents(path):
    files = [f for f in os.listdir(documents_path)]
    documents = []
    for f in files[:10]:    #parsing only 10 xml files for fast execution 
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

def get_stop_words(corpus, alpha = 0.01):
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    return stop_words


def tokenize(document):
    import re

    tokens = []
    
    for string in document.split():
            word=re.sub('[^a-zA-Z0-9]', '', string).lower()
            if word:       #remove empty string
                tokens.append(word)
    return tokens

def stem(document):
    stemmer = PorterStemmer()
    # Perform stemming on the doc
    stemmed_doc = [stemmer.stem(word) for word in document]
    return stemmed_doc

documents_path = r'D:/sem_8/Information Retrieval/pan18-author-profiling-training-dataset-2018-02-27/pan18-author-profiling-training-dataset-2018-02-27/en/text'
documents=get_documents(documents_path)

corpus = [tokenize(document) for document in documents]
#corpus = [nltk.word_tokenize(document) for document in documents]
# print("\n\nAfter tokenizing:\n\n",corpus[0])
N = len(corpus)

stop_words = get_stop_words(corpus)
for i in range(len(corpus)):
    print("original ",len(corpus[i]))
    corpus[i] = [term for term in corpus[i] if term not in stop_words]
    print("stop w ",len(corpus[i]))
    corpus[i] = stem(corpus[i])
    print("stem ",len(corpus[i]))
# print(stop_words)
# print("\n\nAfter stop words removal and stemming:\n\n",corpus[0])

