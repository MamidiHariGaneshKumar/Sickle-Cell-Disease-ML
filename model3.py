import numpy as np
from PIL import Image
import os
import tenseal as ts
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

# Function to load and flatten images
def load_and_flatten_images(path, label):
    """Load images from a directory, convert to grayscale, resize, and flatten"""
    images = []
    labels = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            img = img.resize((16, 16))  # Resize to 16x16
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

# Function to load and preprocess a single image for testing
def load_and_preprocess_single_image(image_path):
    """Load a single image, convert to grayscale, resize, and flatten"""
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((16, 16))  # Resize to match training images
        img_array = np.array(img).flatten()
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Encrypt data function
def encrypt_data(context, data):
    """Encrypt the input data using the provided CKKS encryption context"""
    encrypted_data = []
    for sample in data:
        scaled_sample = (sample / 255.0).astype(np.float64) * 0.1
        try:
            encrypted_vector = ts.ckks_vector(context, scaled_sample.tolist())
            encrypted_data.append(encrypted_vector)
        except Exception as e:
            print(f"Error encrypting sample: {e}")
    return encrypted_data

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy"""
    return np.mean(y_true == y_pred) * 100

# Encrypted Logistic Regression Model
class EncryptedLogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=50):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def safe_sigmoid_approx(self, x):
        """Improved sigmoid approximation using piecewise linear function"""
        try:
            half = ts.ckks_vector(self.context, [0.5])
            linear = ts.ckks_vector(self.context, [0.15])  # Adjusted slope
            return half + x * linear
        except Exception as e:
            print(f"Error in sigmoid approximation: {e}")
            raise
    
    def fit(self, X_encrypted, y, context):
        """Train the model using encrypted data"""
        try:
            self.context = context
            num_samples = len(X_encrypted)
            num_features = len(X_encrypted[0].decrypt())
            
            # Initialize small random weights
            self.weights = np.random.randn(num_features) * 0.01
            self.bias = 0.0
            
            print("Starting encrypted training...")
            batch_size = 10  
            
            for iteration in range(self.num_iterations):
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
                
                weights_encrypted = ts.ckks_vector(context, self.weights.tolist())
                weight_gradients = np.zeros(num_features)
                bias_gradient = 0.0
                
                for i in range(0, num_samples, batch_size):
                    batch_end = min(i + batch_size, num_samples)
                    batch_gradients = np.zeros(num_features)
                    batch_bias = 0.0
                    
                    for j in range(i, batch_end):
                        try:
                            z = X_encrypted[j].dot(weights_encrypted)
                            pred_encrypted = self.safe_sigmoid_approx(z)
                            prediction = float(pred_encrypted.decrypt()[0])
                            error = prediction - y[j]
                            
                            x_dec = np.array(X_encrypted[j].decrypt()) * 0.1
                            batch_gradients += error * x_dec
                            batch_bias += error
                        except Exception as e:
                            print(f"Error processing sample {j}: {e}")
                    
                    weight_gradients += batch_gradients
                    bias_gradient += batch_bias
                
                # Update weights
                self.weights -= self.learning_rate * (weight_gradients / num_samples)
                self.bias -= self.learning_rate * (bias_gradient / num_samples)
                
        except Exception as e:
            print(f"Error in training: {e}")
            raise
    
    def predict(self, X_encrypted):
        """Make predictions on encrypted data"""
        predictions = []
        weights_encrypted = ts.ckks_vector(self.context, self.weights.tolist())
        
        for x_enc in X_encrypted:
            try:
                z = x_enc.dot(weights_encrypted)
                pred_encrypted = self.safe_sigmoid_approx(z)
                pred_value = float(pred_encrypted.decrypt()[0])
                predictions.append(1 if pred_value > 0.5 else 0)
            except Exception as e:
                print(f"Prediction error: {e}")
                predictions.append(0)
        
        return np.array(predictions)

# Create CKKS encryption context
def create_ckks_context():
    """Create the CKKS context for encryption"""
    try:
        params = {
            'scheme': ts.SCHEME_TYPE.CKKS,
            'poly_modulus_degree': 8192,
            'coeff_mod_bit_sizes': [40, 20, 20, 40]
        }
        context = ts.context(**params)
        context.global_scale = 2**20  
        context.generate_galois_keys()
        return context
    except Exception as e:
        print(f"Error creating context: {e}")
        raise

# Main function
def main():
    try:
        # Load dataset
        positive_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Positive"
        negative_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Negative"
        
        X_pos, y_pos = load_and_flatten_images(positive_path, 1)
        X_neg, y_neg = load_and_flatten_images(negative_path, 0)
        
        X = np.array(X_pos + X_neg)
        y = np.array(y_pos + y_neg)

        max_samples = 50
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        context = create_ckks_context()

        X_train_encrypted = encrypt_data(context, X_train)
        X_test_encrypted = encrypt_data(context, X_test)

        model = EncryptedLogisticRegression(learning_rate=0.001, num_iterations=100)
        model.fit(X_train_encrypted, y_train, context)

        y_pred = model.predict(X_test_encrypted)
        accuracy = calculate_accuracy(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}%")

        # Load a new image for prediction
        test_image_path = "C:/Users/mamid/OneDrive/Desktop/test_image.jpg"
        new_image = load_and_preprocess_single_image(test_image_path)

        if new_image is not None:
            new_image_scaled = (new_image / 255.0).astype(np.float64) * 0.1
            encrypted_image = ts.ckks_vector(context, new_image_scaled.tolist())
            encrypted_prediction = model.predict([encrypted_image])
            decrypted_prediction = "Positive (Sickle Cell)" if encrypted_prediction[0] == 1 else "Negative (Healthy)"
            
            plt.imshow(new_image.reshape(16, 16), cmap='gray')
            plt.title(f"Prediction: {decrypted_prediction}")
            plt.axis('off')
            plt.show()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
