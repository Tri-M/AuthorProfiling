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


