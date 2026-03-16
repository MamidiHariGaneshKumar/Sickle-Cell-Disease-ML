import numpy as np
from PIL import Image
import os
import tenseal as ts
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

def load_and_flatten_images(path, label):
    """Load images from directory, convert to grayscale, resize, and flatten"""
    images = []
    labels = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            img = img.resize((16, 16))  # Reduced further to 16x16
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

def encrypt_data(context, data):
    """Encrypt the input data using the provided context"""
    encrypted_data = []
    for sample in data:
        # Normalize to [0, 1] and scale to reduce magnitude
        scaled_sample = (sample / 255.0).astype(np.float64) * 0.1
        try:
            encrypted_vector = ts.ckks_vector(context, scaled_sample.tolist())
            encrypted_data.append(encrypted_vector)
        except Exception as e:
            print(f"Error encrypting sample: {e}")
    return encrypted_data

def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy"""
    return np.mean(y_true == y_pred) * 100

class EncryptedLogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=50):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def safe_sigmoid_approx(self, x):
        """Improved sigmoid approximation using piecewise linear function"""
        try:
            # Piecewise linear approximation
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
            print(f"Number of samples: {num_samples}")
            num_features = len(X_encrypted[0].decrypt())
            print(f"Number of features: {num_features}")
            
            # Initialize with small weights
            self.weights = np.random.randn(num_features) * 0.01
            self.bias = 0.0
            
            print("Starting encrypted training...")
            batch_size = 10  # Increased batch size
            
            for iteration in range(self.num_iterations):
                print(f"Starting iteration {iteration + 1}/{self.num_iterations}")
                
                weights_encrypted = ts.ckks_vector(context, self.weights.tolist())
                weight_gradients = np.zeros(num_features)
                bias_gradient = 0.0
                
                for i in range(0, num_samples, batch_size):
                    batch_end = min(i + batch_size, num_samples)
                    
                    batch_gradients = np.zeros(num_features)
                    batch_bias = 0.0
                    
                    for j in range(i, batch_end):
                        try:
                            # Compute dot product
                            z = X_encrypted[j].dot(weights_encrypted)
                            
                            # Apply safe sigmoid approximation
                            pred_encrypted = self.safe_sigmoid_approx(z)
                            prediction = float(pred_encrypted.decrypt()[0])
                            
                            # Compute error
                            error = prediction - y[j]
                            
                            # Decrypt input for gradient computation
                            x_dec = np.array(X_encrypted[j].decrypt()) * 0.1
                            batch_gradients += error * x_dec
                            batch_bias += error
                            
                        except Exception as e:
                            print(f"Error processing sample {j}: {e}")
                            continue
                    
                    # Accumulate and update gradients
                    weight_gradients += batch_gradients
                    bias_gradient += batch_bias
                
                # Average and update
                weight_gradients /= num_samples
                bias_gradient /= num_samples
                
                # Update weights and bias
                self.weights -= self.learning_rate * weight_gradients
                self.bias -= self.learning_rate * bias_gradient
                
                print(f"Completed iteration {iteration + 1}, Bias gradient: {bias_gradient}")
                
        except Exception as e:
            print(f"Error in training: {e}")
            raise
    
    def predict(self, X_encrypted):
        """Make predictions on encrypted data"""
        predictions = []
        weights_encrypted = ts.ckks_vector(self.context, self.weights.tolist())
        
        for x_enc in X_encrypted:
            try:
                # Compute dot product
                z = x_enc.dot(weights_encrypted)
                
                # Safe sigmoid approximation
                pred_encrypted = self.safe_sigmoid_approx(z)
                pred_value = float(pred_encrypted.decrypt()[0])
                
                predictions.append(1 if pred_value > 0.5 else 0)
            except Exception as e:
                print(f"Prediction error: {e}")
                predictions.append(0)  # Default to negative prediction
        
        return np.array(predictions)

def create_ckks_context():
    """Create the CKKS context with optimized parameters"""
    try:
        params = {
            'scheme': ts.SCHEME_TYPE.CKKS,
            'poly_modulus_degree': 8192,
            'coeff_mod_bit_sizes': [40, 20, 20, 40]
        }
        context = ts.context(**params)
        context.global_scale = 2**20  # Reduced global scale
        context.generate_galois_keys()
        return context
    except Exception as e:
        print(f"Error creating context: {e}")
        raise

def main():
    try:
        # Load data
        positive_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Positive"
        negative_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Negative"
        
        print("Loading images...")
        X_pos, y_pos = load_and_flatten_images(positive_path, 1)
        X_neg, y_neg = load_and_flatten_images(negative_path, 0)
        
        X = np.array(X_pos + X_neg)
        y = np.array(y_pos + y_neg)
        
        print(f"Total samples: {len(X)}")
        print(f"Input shape: {X[0].shape}")
        
        # Take a smaller subset of data for testing
        max_samples = 50  # Increased for better training
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"Reduced to {max_samples} samples for testing")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Creating encryption context...")
        context = create_ckks_context()
        
        print("Encrypting training data...")
        start_time = time.time()
        X_train_encrypted = encrypt_data(context, X_train)
        X_test_encrypted = encrypt_data(context, X_test)
        print(f"Encryption time: {time.time() - start_time:.2f} seconds")
        
        print("Training model on encrypted data...")
        model = EncryptedLogisticRegression(learning_rate=0.001, num_iterations=100)
        start_time = time.time()
        model.fit(X_train_encrypted, y_train, context)
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        
        print("Making predictions...")
        y_pred = model.predict(X_test_encrypted)
        accuracy = calculate_accuracy(y_test, y_pred)
        print(f"Encrypted Logistic Regression Accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()