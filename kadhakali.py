# import cv2
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import pickle

# # Define the label names
# label_names = {
#     1: "Pataka",
#     2: "Tripataka",
#     3: "Ardhapataka",
#     4: "Kartari Mukha",
#     5: "Mayura",
#     6: "Ardhachandra",
#     7: "Mushti",
#     8: "Shikhara",
#     9: "Kapittha",
#     10: "Katamukha",
#     11: "Sarpasirsha",
#     12: "Mrigashirsha",
#     13: "Simhamukha",
#     14: "Kangula",
#     15: "Alapadma",
#     16: "Chatura",
#     17: "Bhramara",
#     18: "Hamsasya",
#     19: "Hamsapaksha",
#     20: "Samdamsha",
#     21: "Mukula",
#     22: "Tamrachuda",
#     23: "Trishula",
#     24: "Pasham"
# }

# def read_csv_file(file_path):
#     try:
#         data = pd.read_csv(file_path)
#         return data
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#         return None

# def extract_data(csv_file):
#     ids = csv_file['id'].tolist()
#     labels = csv_file['label'].tolist() if 'label' in csv_file.columns else None
#     return ids, labels


# def load_images(image_folder, image_ids):
#     images = []
#     for image_id in image_ids:
#         image_path = os.path.join(image_folder, str(image_id) + '.jpg')
#         if not os.path.exists(image_path):
#             print(f"Image not found: {image_path}")
#             continue

#         try:
#             image = cv2.imread(image_path)
#             if image is None:
#                 print(f"Error loading image {image_path}: Invalid image file")
#                 continue
#             images.append(image)
#         except Exception as e:
#             print(f"Error loading image {image_path}: {str(e)}")
#     return images


# def preprocess_images(images):
#     preprocessed_images = []
#     for image in images:
#         # Flatten the image dimensions
#         flattened_image = image.flatten()
#         preprocessed_images.append(flattened_image)
#     return preprocessed_images


# def train_model(train_ids, train_labels):
#     # Load train images
#     train_image_folder = 'images'
#     train_images = load_images(train_image_folder, train_ids)
#     print("Train images:", train_images)

#     # Preprocess train images
#     preprocessed_train_images = preprocess_images(train_images)

#     # Split train data into train and validation sets
#     train_images, val_images, train_labels, val_labels = train_test_split(
#         preprocessed_train_images, train_labels, test_size=0.2, random_state=42
#     )
#     print("Train images:", train_images)
#     print("Validation images:", val_images)

#     # Train a model (e.g., Support Vector Machine)
#     model = SVC()
#     model.fit(train_images, train_labels)

#     # Save the trained model
#     with open('hand_gesture_model.pkl', 'wb') as file:
#         pickle.dump(model, file)

#     # Evaluate model performance on validation set
#     val_predictions = model.predict(val_images)
#     val_accuracy = accuracy_score(val_labels, val_predictions)
#     print("Validation Accuracy:", val_accuracy)

#     return model

# # Read train.csv
# train_file_path = 'train_485.csv'
# train_data = read_csv_file(train_file_path)
# if train_data is not None:
#     train_ids, train_labels = extract_data(train_data)
#     print("Train data:")
#     print("IDs:", train_ids)
#     print("Labels:", train_labels)

#     if train_labels is not None:
#         # Train the model and save it
#         model = train_model(train_ids, train_labels)
#         print("Model trained and saved as hand_gesture_model.pkl")
#     else:
#         print("No labels found in the train data.")
# else:
#     print("Error reading train.csv file.")

import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from sklearn.feature_selection import SelectKBest, f_classif

# Define the label names
label_names = {
    1: "Pataka",
    2: "Tripataka",
    3: "Ardhapataka",
    4: "Kartari Mukha",
    5: "Mayura",
    6: "Ardhachandra",
    7: "Mushti",
    8: "Shikhara",
    9: "Kapittha",
    10: "Katamukha",
    11: "Sarpasirsha",
    12: "Mrigashirsha",
    13: "Simhamukha",
    14: "Kangula",
    15: "Alapadma",
    16: "Chatura",
    17: "Bhramara",
    18: "Hamsasya",
    19: "Hamsapaksha",
    20: "Samdamsha",
    21: "Mukula",
    22: "Tamrachuda",
    23: "Trishula",
    24: "Pasham"
}

def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def extract_data(csv_file):
    ids = csv_file['id'].tolist()
    labels = csv_file['label'].tolist() if 'label' in csv_file.columns else None
    return ids, labels

def load_images(image_folder, image_ids):
    images = []
    valid_indices = []  # Store the indices of successfully loaded images
    for i, image_id in enumerate(image_ids):
        image_path = os.path.join(image_folder, str(image_id) + '.jpg')
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}: Invalid image file")
                continue
            images.append(image)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
    return images, valid_indices

def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Resize the image to desired dimensions
        resized_image = cv2.resize(image, (128, 94))

        # Flatten the image dimensions
        flattened_image = resized_image.flatten()
        preprocessed_images.append(flattened_image)
    return preprocessed_images

def train_model(train_ids, train_labels, max_features):
    # Load train images
    train_image_folder = 'images'
    train_images, valid_indices = load_images(train_image_folder, train_ids)
    train_labels = [train_labels[i] for i in valid_indices]  # Update train_labels accordingly

    # Preprocess train images
    preprocessed_train_images = preprocess_images(train_images)

    # Select top k features
    feature_selector = SelectKBest(score_func=f_classif, k=max_features)
    train_images = feature_selector.fit_transform(preprocessed_train_images, train_labels)

    # Split train data into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    print("Train images:", len(train_images))
    print("Validation images:", len(val_images))

    # Train a model (e.g., Support Vector Machine) with max_features
    model = SVC()
    model.fit(train_images, train_labels)

    # Save the trained model
    with open('hand_gesture_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Evaluate model performance on validation set
    val_predictions = model.predict(val_images)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print("Validation Accuracy:", val_accuracy)

    return model

# Read train.csv
train_file_path = 'train_485.csv'
train_data = read_csv_file(train_file_path)
if train_data is not None:
    train_ids, train_labels = extract_data(train_data)
    print("Train data:")
    print("IDs:", train_ids)
    print("Labels:", train_labels)

    if train_labels is not None:
        # Set the desired number of features
        max_features = 36096
        # Train the model and save it
        model = train_model(train_ids, train_labels, max_features)
        print("Model trained and saved as hand_gesture_model.pkl")
    else:
        print("No labels found in the train data.")
else:
    print("Error reading train.csv file.")
