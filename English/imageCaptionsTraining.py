def extract_image_captions(filename):
    user_captions = {}
    current_user = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            if line.endswith(':'):
                current_user = line[:-1]  # Extract user ID
                user_captions[current_user] = {}
            elif '.jpeg caption:' in line:
                parts = line.split('.jpeg caption:')
                image_id = parts[0].strip()
                caption = parts[1].split(',')[0].strip()
                user_captions[current_user][image_id] = caption
    
    return user_captions

def write_captions_to_file(user_captions, output_filename):
    with open(output_filename, 'w') as f:
        for user, image_captions in user_captions.items():
            if user != 'photo':  # Skip writing the folder name line
                f.write(f"User: {user}\n")
                for image_id, caption in image_captions.items():
                    f.write(f"{image_id}: {caption}\n")
                f.write("\n")  # Add a blank line between users

# Example usage:
caption_filename = 'captions_output.txt'
output_filename = 'extracted_captions.txt'

# Extract image captions from the file
image_captions_by_user = extract_image_captions(caption_filename)

# Write the extracted captions to a file
write_captions_to_file(image_captions_by_user, output_filename)

print(f"Extracted captions have been written to '{output_filename}'.")

import re
from sklearn.feature_extraction.text import TfidfVectorizer

def load_captions(filename):
    user_captions = {}
    current_user = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            # Match user line
            user_match = re.match(r'^User: (.+)$', line.strip())
            if user_match:
                current_user = user_match.group(1)
                user_captions[current_user] = []
            else:
                # Match image caption lines
                caption_match = re.match(r'^(\S+): (.+)$', line.strip())
                if caption_match:
                    image_id = caption_match.group(1)
                    caption = caption_match.group(2)
                    user_captions[current_user].append((image_id, caption))
    
    return user_captions

def preprocess_caption(caption):
    # Convert to lowercase and remove punctuation
    processed_caption = caption.lower()
    processed_caption = re.sub(r'[^\w\s]', '', processed_caption)
    return processed_caption.strip()

def perform_tfidf_vectorization(user_captions):
    tfidf_vectors_per_user = {}
    
    for user, captions in user_captions.items():
        processed_captions = [preprocess_caption(caption) for _, caption in captions]
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        
        # Fit and transform the processed captions into TF-IDF vectors
        tfidf_vectors = tfidf_vectorizer.fit_transform(processed_captions)
        
        # Store the TF-IDF vectorizer and vectors in a dictionary
        tfidf_vectors_per_user[user] = (tfidf_vectorizer, tfidf_vectors)
    
    return tfidf_vectors_per_user

def print_tfidf_vectors(tfidf_vectors_per_user):
    for user, (tfidf_vectorizer, tfidf_vectors) in tfidf_vectors_per_user.items():
        print(f"User: {user}")
        for i, (_, caption) in enumerate(user_captions[user]):
            print(f"{user}.{i}: {caption}")
        print("TF-IDF Vectors:")
        print(tfidf_vectors.toarray())  # Print TF-IDF vectors as arrays
        print()


# Define the filename containing the captions
caption_filename = 'extracted_captions.txt'

# Load and parse image captions for each user
user_captions = load_captions(caption_filename)

# Perform TF-IDF vectorization for each user
tfidf_vectors_per_user = perform_tfidf_vectorization(user_captions)

def print_tfidf_scores(tfidf_vectors_per_user):
    for user, (tfidf_vectorizer, tfidf_vectors) in tfidf_vectors_per_user.items():
        print(f"User: {user}")
        for i, (_, caption) in enumerate(user_captions[user]):
            print(f"{user}.{i}: {caption}")
        
        # Get the feature names (words) from the TF-IDF vectorizer
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Retrieve the TF-IDF scores for each term (word) in the vocabulary
        tfidf_scores = tfidf_vectors.toarray()
        
        # Print the TF-IDF scores for each term (word)
        for j, term in enumerate(feature_names):
            print(f"TF-IDF Score for '{term}': {tfidf_scores[:, j]}")
        print()

# Call the function to print TF-IDF scores for each user
print_tfidf_scores(tfidf_vectors_per_user)

# Print TF-IDF vectors for each user
# print_tfidf_vectors(tfidf_vectors_per_user)

