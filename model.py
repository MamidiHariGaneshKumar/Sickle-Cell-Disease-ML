import cv2
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tenseal as ts

# Folders for positive and negative samples
positive_folder = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Positive"
negative_folder = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Negative"

# Function to load and preprocess the images
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize image to 64x64
            images.append(img.flatten())  # Flatten the image to a vector
            labels.append(1 if folder == positive_folder else 0)  # Label 1 for Positive, 0 for Negative
    return images, labels

# Load positive and negative images
positive_images, positive_labels = load_images_from_folder(positive_folder)
negative_images, negative_labels = load_images_from_folder(negative_folder)

# Combine the data and labels
X = np.array(positive_images + negative_images)
y = np.array(positive_labels + negative_labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setup CKKS encryption context
def setup_ckks_encryption():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 60])
    context.generate_galois_keys()  # Generate keys for encryption and decryption
    return context

# Encrypt the data
def encrypt_data(context, data):
    encryptor = context.encryptor()
    encrypted_data = [encryptor.encrypt([value]) for value in data]
    return encrypted_data

# Train a Logistic Regression model on decrypted data
def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

# Train the model on encrypted data
def train_model_on_encrypted_data(context, X_train, y_train):
    encrypted_X_train = encrypt_data(context, X_train.flatten())
    encrypted_model = LogisticRegression(max_iter=1000)
    # Placeholder for homomorphic training. You would typically train on encrypted data.
    encrypted_model.fit(encrypted_X_train, y_train)
    return encrypted_model

# Decrypt the predictions
def decrypt_predictions(context, encrypted_predictions):
    decryptor = context.decryptor()
    decrypted_predictions = [decryptor.decrypt(value)[0] for value in encrypted_predictions]
    return decrypted_predictions

# Main execution
context = setup_ckks_encryption()  # Setup CKKS encryption context

# You can either train on decrypted data or encrypted data
model = train_logistic_regression(X_train, y_train)  # Train on decrypted data

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"Logistic Regression Accuracy: {accuracy:.2f}%")
