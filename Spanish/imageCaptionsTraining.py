import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def load_captions(filename):
    user_captions = {}
    current_user = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            # Match user line
            user_match = re.match(r'^(\S+):$', line.strip())
            if user_match:
                current_user = user_match.group(1)
                user_captions[current_user] = []
            else:
                # Match image caption lines
                caption_match = re.match(r'^(\S+\.?\d*\.(jpeg|jpg|png)) caption: (.+), \d+\.\d+%$', line.strip())
                if caption_match:
                    image_id = caption_match.group(1)
                    caption = caption_match.group(3)
                    user_captions[current_user].append((image_id, caption))
    
    return user_captions

def preprocess_caption(caption):
    # Convert to lowercase and remove punctuation
    processed_caption = caption.lower()
    processed_caption = re.sub(r'[^\w\s]', '', processed_caption)
    return processed_caption.strip()

def perform_tfidf_lsa(user_captions):
    lsa_vectors_per_user = {}
    
    for user, captions in user_captions.items():
        processed_captions = [preprocess_caption(caption) for _, caption in captions]
        
        # Debug: Print processed captions
        print(f"Processed Captions for User {user}:")
        for caption in processed_captions:
            print(caption)
        
        # Filter out empty or insufficient captions
        filtered_captions = [caption for caption in processed_captions if caption]  # Filter out empty captions
        
        if not filtered_captions:
            print(f"Warning: No valid captions found for User {user}. Skipping...")
            continue
        
        # Create TF-IDF vectorizer with appropriate settings
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        
        # Fit and transform the processed captions into TF-IDF vectors
        tfidf_vectors = tfidf_vectorizer.fit_transform(filtered_captions)
        
        # Determine a suitable number of components for TruncatedSVD (LSA)
        n_components = min(tfidf_vectors.shape) - 1
        if n_components < 2:
            print(f"Warning: Insufficient features ({tfidf_vectors.shape[1]}) for LSA. Skipping user {user}.")
            continue
        
        # Perform Truncated SVD (LSA)
        lsa = TruncatedSVD(n_components=n_components)
        lsa_vectors = lsa.fit_transform(tfidf_vectors)
        
        # Store the LSA vectors in a dictionary
        lsa_vectors_per_user[user] = (lsa, lsa_vectors)
    
    return lsa_vectors_per_user

def write_lsa_results_to_file(lsa_vectors_per_user, output_filename):
    with open(output_filename, 'w') as f:
        for user, (lsa, lsa_vectors) in lsa_vectors_per_user.items():
            f.write(f"User: {user}\n")
            for component_idx in range(lsa_vectors.shape[1]):
                f.write(f"LSA Component {component_idx + 1}:\n")
                for value_idx, value in enumerate(lsa.components_[component_idx]):
                    f.write(f"Value {value_idx + 1}: {value:.12f}\n")
            f.write("\n")

# Define the filename containing the captions
caption_filename = r'F:\AP\AuthorProfiling\Spanish\spanish_captions_output.txt'
output_filename = r'spanish_captions_output_lsa_components.txt'

# Load and parse image captions for each user
user_captions = load_captions(caption_filename)

# Perform TF-IDF vectorization and LSA
lsa_vectors_per_user = perform_tfidf_lsa(user_captions)

# Write LSA results to a file
write_lsa_results_to_file(lsa_vectors_per_user, output_filename)

print(f"LSA results have been written to '{output_filename}'.")
