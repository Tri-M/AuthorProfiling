import xml.etree.ElementTree as ET
import re
import nltk
from nltk.corpus import stopwords

import os
import xml.etree.ElementTree as ET
import re

nltk.download('stopwords')

# Load Spanish stopwords
stop_words = set(stopwords.words('spanish'))

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
    text = re.sub(r'[:;]-?[)D]', '', text)
    text = re.sub(r'[:;]-?\(', '', text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    text = re.sub(r'RT\b', '', text)
    
    text = re.sub(r'#\w+', '', text)
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    return text.strip()


folder_path = 'F:/pan18-author-profiling-training-dataset-2018-02-27/pan18-author-profiling-training-dataset-2018-02-27/es/text'

output_file_path = 'spanish_output1.txt'

preprocessed_data = preprocess_xml_files(folder_path)


with open(output_file_path, 'w', encoding='utf-8') as f:
    for idx, author_data in enumerate(preprocessed_data):
        f.write(f"User {idx + 1}: {author_data['lang']}\n")
        for i, tweet in enumerate(author_data['documents'], start=1):
            f.write(f"Tweet {i}: {tweet}\n\n")

print("Output written to", output_file_path)