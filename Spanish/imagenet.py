# Import the necessary packages
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import VGG16
import numpy as np
import cv2
import os


def process_images_in_folder(folder_path, model, output_file):
    count_processed = 0  # Initialize a counter for processed images
    
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            subfolder_name = os.path.basename(root)
            f.write(f"{subfolder_name}:\n")

            for filename in files:
                if filename.endswith(('.jpeg', '.jpg', '.png')):  # Check for image file extensions
                    img_path = os.path.join(root, filename)

                    try:
                        orig = cv2.imread(img_path)

                        if orig is None:
                            print(f"Error loading image: {img_path}")
                            continue  # Skip to the next image if loading fails

                        height, width = orig.shape[:2]
                        aspect_ratio = width / height
                        if width > height:
                            new_width = 350
                            new_height = int(new_width / aspect_ratio)
                        else:
                            new_height = 350
                            new_width = int(new_height * aspect_ratio)
                        resized_image = cv2.resize(orig, (new_width, new_height))

                        print("[INFO] Loading and preprocessing image...")
                        image = image_utils.load_img(img_path, target_size=(224, 224))
                        image = image_utils.img_to_array(image)
                        image = np.expand_dims(image, axis=0)
                        image = preprocess_input(image)

                        print("[INFO] Classifying image...")
                        preds = model.predict(image)
                        P = decode_predictions(preds)

                        (imagenetID, label, prob) = P[0][0]
                        caption = f"{filename} caption: {label}, {prob * 100:.2f}%"
                        f.write(f"{caption}\n")

                        count_processed += 1  # Increment the processed image counter

                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
                        continue

    return count_processed


folder_path = r"F:\pan18-author-profiling-training-dataset-2018-02-27\pan18-author-profiling-training-dataset-2018-02-27\es\photo"
output_file = "spanish_captions_output.txt"

print("[INFO] Loading network...")
model = VGG16(weights="imagenet")

num_processed = process_images_in_folder(folder_path, model, output_file)

print(f"Number of images processed: {num_processed}")
print(f"Captions saved to: {output_file}")