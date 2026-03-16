import numpy as np
from PIL import Image
import os
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create context for CKKS encryption
def create_ckks_context():
    # Increased polynomial modulus degree to handle larger input size
    poly_modulus_degree = 32768  # Increased from 8192
    coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 60]  # Added more coefficient moduli
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

# Encrypt data using CKKS with batch processing
def encrypt_data(context, data):
    encrypted_data = []
    batch_size = 1000  # Process in smaller batches
    
    for sample in data:
        try:
            # Scale and normalize the data
            scaled_sample = (sample / 255.0).astype(np.float64)
            # Create encrypted vector
            encrypted_vector = ts.ckks_vector(context, scaled_sample.tolist())
            encrypted_data.append(encrypted_vector)
        except Exception as e:
            print(f"Error encrypting sample: {str(e)}")
            continue
            
    return encrypted_data

# Decrypt data
def decrypt_data(encrypted_data):
    decrypted_data = []
    for enc_vector in encrypted_data:
        try:
            dec_vector = enc_vector.decrypt()
            decrypted_data.append(dec_vector)
        except Exception as e:
            print(f"Error decrypting vector: {str(e)}")
            continue
    return np.array(decrypted_data)

# Load and flatten images with error handling
def load_and_flatten_images(path, label):
    images = []
    labels = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        try:
            img = Image.open(img_path)
            # Convert to grayscale if image is in color
            if img.mode != 'L':
                img = img.convert('L')
            img = img.resize((64, 64))
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    return images, labels

def main():
    try:
        # Paths to your datasets
        positive_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Positive"
        negative_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Negative"

        # Load data
        print("Loading images...")
        X_pos, y_pos = load_and_flatten_images(positive_path, 1)
        X_neg, y_neg = load_and_flatten_images(negative_path, 0)

        X = np.array(X_pos + X_neg)
        y = np.array(y_pos + y_neg)

        print(f"Total samples: {len(X)}")
        print(f"Input shape: {X[0].shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create CKKS context
        print("Creating encryption context...")
        context = create_ckks_context()

        # Encrypt training and test data
        print("Encrypting training data...")
        X_train_encrypted = encrypt_data(context, X_train)
        print("Encrypting test data...")
        X_test_encrypted = encrypt_data(context, X_test)

        # Decrypt data for training
        print("Decrypting data for training...")
        X_train_decrypted = decrypt_data(X_train_encrypted)
        X_test_decrypted = decrypt_data(X_test_encrypted)

        # Train model
        print("Training model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_decrypted, y_train)

        # Make predictions
        y_pred = model.predict(X_test_decrypted)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Logistic Regression Accuracy with HE: {accuracy * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()