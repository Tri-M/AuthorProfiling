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


# Example usage:
caption_filename = 'captions_output.txt'
output_filename = 'extracted_captions.txt'

# Extract image captions from the file
image_captions_by_user = extract_image_captions(caption_filename)

# Write the extracted captions to a file
write_captions_to_file(image_captions_by_user, output_filename)

print(f"Extracted captions have been written to '{output_filename}'.")


#TF IDF VECTORIZATION
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_captions(filename):
    user_captions = {}
    current_user = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                current_user = line[:-1]  # Extract the user ID
                if current_user not in user_captions:
                    user_captions[current_user] = []  # Initialize list for captions
            elif line:  # Process caption lines
                parts = line.split(': ')
                if len(parts) > 1:
                    caption_info = parts[-1].rsplit(',', 1)
                    caption = caption_info[0].strip()
                    user_captions[current_user].append(caption)  # Store image caption
    
    return user_captions


def preprocess_and_vectorize_captions(user_captions):
    tfidf_vectors_per_user = {}
    stop_words = []  # Add custom stop words if needed
    
    for user, captions in user_captions.items():
        # Preprocess captions: lowercase and remove punctuation
        processed_captions = []
        for caption in captions:
            # Apply preprocessing steps
            processed_caption = preprocess_caption(caption)
            if processed_caption:  # Check if the processed caption is not empty
                processed_captions.append(processed_caption)
        
        if processed_captions:
            # Create TF-IDF vectorizer with custom options
            tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
            
            try:
                # Fit and transform the captions into TF-IDF vectors
                tfidf_vectors = tfidf_vectorizer.fit_transform(processed_captions)
                tfidf_vectors_per_user[user] = (tfidf_vectorizer, tfidf_vectors)
            except ValueError as e:
                # Handle specific errors (e.g., empty vocabulary)
                print(f"Error processing user {user}: {e}")
        else:
            print(f"No valid captions found for user {user}")
    
    return tfidf_vectors_per_user

def preprocess_caption(caption):
    # Example preprocessing steps (customize based on your needs)
    processed_caption = caption.lower()  # Convert to lowercase
    processed_caption = remove_punctuation(processed_caption)  # Remove punctuation
    
    return processed_caption.strip()  # Remove leading/trailing spaces

def remove_punctuation(text):
    # Example function to remove punctuation (customize based on your needs)
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Example usage:
caption_filename = 'extracted_captions.txt'

# Load image captions from the file
user_captions = load_captions(caption_filename)

# Preprocess and vectorize captions for each user
tfidf_vectors_per_user = preprocess_and_vectorize_captions(user_captions)


# Print the loaded captions for each user
for user, captions in user_captions.items():
    print(f"User: {user}")
    for i, caption in enumerate(captions):
        print(f"{user}.{i}: {caption}")




def generate_tfidf_vectors(user_captions):
    tfidf_vectors_by_user = {}
    for user, captions in user_captions.items():
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        
        # Compute TF-IDF vectors for each caption
        tfidf_matrix = tfidf_vectorizer.fit_transform(captions)
        
        # Store TF-IDF vectorizer and matrix for the user
        tfidf_vectors_by_user[user] = {
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix
        }
    
    return tfidf_vectors_by_user

# Example usage:
caption_filename = 'extracted_captions.txt'

# Load image captions from the file
# user_captions = load_captions(caption_filename)

# Generate TF-IDF vectors for each user's image captions
tfidf_vectors_by_user = generate_tfidf_vectors(user_captions)

# Now tfidf_vectors_by_user contains TF-IDF vectors for each user's image captions
# Access them using user IDs and process further as needed
for user, tfidf_data in tfidf_vectors_by_user.items():
    print(f"User: {user}")
    print("TF-IDF Vectors:")
    tfidf_matrix = tfidf_data['tfidf_matrix']
    print(tfidf_matrix.toarray())  # Convert sparse matrix to array for display

# You can further process or save these TF-IDF vectors as needed
