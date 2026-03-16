import numpy as np
from PIL import Image
import os
import tenseal as ts
from sklearn.model_selection import train_test_split, StratifiedKFold
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
            img = img.resize((16, 16))  # Reduced to 16x16
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

def load_single_image(image_path):
    """Load and preprocess a single image for prediction"""
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((16, 16))  # Same size as training images
        img_array = np.array(img).flatten()
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

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
        self.context = None
        self.training_history = {'loss': [], 'accuracy': []}
    
    def safe_sigmoid_approx(self, x):
        """Improved sigmoid approximation using piecewise linear function"""
        try:
            half = ts.ckks_vector(self.context, [0.5])
            linear = ts.ckks_vector(self.context, [0.15])
            return half + x * linear
        except Exception as e:
            print(f"Error in sigmoid approximation: {e}")
            raise
    
    def calculate_metrics(self, X_encrypted, y):
        """Calculate loss and accuracy for current iteration"""
        predictions, raw_predictions = self.predict(X_encrypted)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y) * 100
        
        # Calculate binary cross entropy loss
        epsilon = 1e-15  # Small constant to avoid log(0)
        raw_predictions = np.clip(raw_predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(raw_predictions) + (1 - y) * np.log(1 - raw_predictions))
        
        return loss, accuracy
    
    def fit(self, X_encrypted, y, context):
        """Train the model using encrypted data"""
        try:
            self.context = context
            num_samples = len(X_encrypted)
            num_features = len(X_encrypted[0].decrypt())
            
            # Initialize with small weights
            self.weights = np.random.randn(num_features) * 0.01
            self.bias = 0.0
            
            # Adaptive batch size based on dataset size
            batch_size = min(10, max(1, num_samples // 10))
            
            for iteration in range(self.num_iterations):
                print(f"\nStarting iteration {iteration+1}/{self.num_iterations}")
                
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
                            continue
                    
                    weight_gradients += batch_gradients
                    bias_gradient += batch_bias
                
                # Update weights and bias
                weight_gradients /= num_samples
                bias_gradient /= num_samples
                self.weights -= self.learning_rate * weight_gradients
                self.bias -= self.learning_rate * bias_gradient
                
                # Calculate and store metrics
                loss, accuracy = self.calculate_metrics(X_encrypted, y)
                self.training_history['loss'].append(loss)
                self.training_history['accuracy'].append(accuracy)
                
                # Print metrics in the requested format
                print(f"Iteration {iteration+1}: Accuracy = {accuracy:.2f}%, Loss = {loss:.4f}")
            
        except Exception as e:
            print(f"Error in training: {e}")
            raise
    
    def predict(self, X_encrypted):
        """Make predictions on encrypted data"""
        predictions = []
        raw_predictions = []
        weights_encrypted = ts.ckks_vector(self.context, self.weights.tolist())
        
        for x_enc in X_encrypted:
            try:
                # Compute dot product
                z = x_enc.dot(weights_encrypted)
                
                # Safe sigmoid approximation
                pred_encrypted = self.safe_sigmoid_approx(z)
                pred_value = float(pred_encrypted.decrypt()[0])
                raw_predictions.append(pred_value)
                predictions.append(1 if pred_value > 0.5 else 0)
            except Exception as e:
                print(f"Prediction error: {e}")
                predictions.append(0)  # Default to negative prediction
                raw_predictions.append(0.0)
        
        return np.array(predictions), np.array(raw_predictions)

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

def predict_single_image(model, image_path, context):
    """Make prediction on a single image using the encrypted model"""
    try:
        # Load and preprocess the image
        img_array = load_single_image(image_path)
        if img_array is None:
            return None, None
        
        # Normalize and scale the image data
        scaled_image = (img_array / 255.0).astype(np.float64) * 0.1
        
        # Encrypt the image
        print("Encrypting image...")
        encrypted_image = ts.ckks_vector(context, scaled_image.tolist())
        
        # Make prediction
        print("Making encrypted prediction...")
        prediction, raw_prediction = model.predict([encrypted_image])
        
        return prediction[0], raw_prediction[0]
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

def process_new_image(model, context, image_path):
    """Complete pipeline for processing a new image with visualization"""
    try:
        print(f"Processing image: {image_path}")
        
        # Load original image for display
        original_img = Image.open(image_path)
        
        # Make prediction
        prediction, confidence = predict_single_image(model, image_path, context)
        
        if prediction is not None:
            # Create figure for side-by-side display
            plt.figure(figsize=(12, 5))
            
            # Display original image
            plt.subplot(1, 2, 1)
            plt.imshow(original_img, cmap='gray' if original_img.mode == 'L' else None)
            plt.title('Input Image')
            plt.axis('off')
            
            # Display preprocessed image with prediction
            plt.subplot(1, 2, 2)
            img_array = load_single_image(image_path)
            plt.imshow(img_array.reshape(16, 16), cmap='gray')
            result = "Positive (Sickle Cell)" if prediction == 1 else "Negative (Healthy)"
            plt.title(f"Prediction: {result}\nConfidence: {confidence:.2f}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nPrediction Result: {result}")
            print(f"Confidence Score: {confidence:.4f}")
        else:
            print("Failed to make prediction")
            
    except Exception as e:
        print(f"Error processing image: {e}")

def plot_learning_curves(history):
    """Plot learning curves for accuracy and loss"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], marker='o')
    plt.title('Training Accuracy Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], marker='o', color='r')
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Load dataset
        positive_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Positive"
        negative_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Negative"
        
        print("Loading and processing images...")
        X_pos, y_pos = load_and_flatten_images(positive_path, 1)
        X_neg, y_neg = load_and_flatten_images(negative_path, 0)
        
        X = np.array(X_pos + X_neg)
        y = np.array(y_pos + y_neg)

        print(f"Total samples: {len(X)}")
        print(f"Input shape: {X[0].shape}")
        print(f"Class distribution: Positive: {sum(y)}, Negative: {len(y) - sum(y)}")

        # Ask user if they want to use the full dataset or a subset
        use_full_dataset = input("Use full dataset? (y/n, default: y): ").strip().lower() != 'n'
        
        if not use_full_dataset:
            # If user doesn't want full dataset, use stratified sampling
            max_samples = int(input("Enter maximum number of samples to use (default: 100): ") or "100")
            
            # Ensure we don't try to sample more than available
            max_samples = min(max_samples, len(X))
            
            # Use stratified sampling to maintain class distribution
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-max_samples/len(X), random_state=42)
            for train_idx, _ in sss.split(X, y):
                X = X[train_idx]
                y = y[train_idx]
            
            print(f"Using {len(X)} stratified samples")
            print(f"New class distribution: Positive: {sum(y)}, Negative: {len(y) - sum(y)}")
        
        # Split data with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Creating encryption context...")
        context = create_ckks_context()

        print("Encrypting training data...")
        start_time = time.time()
        X_train_encrypted = encrypt_data(context, X_train)
        X_test_encrypted = encrypt_data(context, X_test)
        print(f"Encryption time: {time.time() - start_time:.2f} seconds")

        # Adjust learning parameters based on dataset size
        if len(X_train) > 100:
            learning_rate = 0.0005  # Lower learning rate for larger datasets
            num_iterations = 150    # More iterations for larger datasets
        else:
            learning_rate = 0.001
            num_iterations = 100

        print(f"Training model with learning_rate={learning_rate}, iterations={num_iterations}")
        model = EncryptedLogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations)
        
        start_time = time.time()
        model.fit(X_train_encrypted, y_train, context)
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")

        # Plot learning curves
        plot_learning_curves(model.training_history)

        print("Making predictions on test set...")
        y_pred, raw_preds = model.predict(X_test_encrypted)
        test_accuracy = calculate_accuracy(y_test, y_pred)
        print(f"Encrypted Logistic Regression Test Accuracy: {test_accuracy:.2f}%")
        
        # Confusion matrix
        TP = sum((y_test == 1) & (y_pred == 1))
        TN = sum((y_test == 0) & (y_pred == 0))
        FP = sum((y_test == 0) & (y_pred == 1))
        FN = sum((y_test == 1) & (y_pred == 0))
        
        print("\nConfusion Matrix:")
        print(f"True Positive: {TP}, True Negative: {TN}")
        print(f"False Positive: {FP}, False Negative: {FN}")
        
        # Precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Process test images with visualization
        while True:
            test_image_path = input("\nEnter the path to a test image (or 'q' to quit): ").strip()
            if test_image_path.lower() == 'q':
                break
            if test_image_path:
                process_new_image(model, context, test_image_path)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()