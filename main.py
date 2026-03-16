import numpy as np
from PIL import Image
import os
import tenseal as ts
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
import io
import base64
import matplotlib.pyplot as plt
import json
from threading import Thread
from queue import Queue
from matplotlib.figure import Figure
from io import BytesIO
import base64

# Global variables to store model and context
model = None
context = None
progress_queue = Queue()

def create_ckks_context():
    """Create and return a TenSEAL context for CKKS encryption"""
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
            img = img.resize((16, 16))
            img_array = np.array(img).flatten() / 255.0  # Normalize here
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

def load_single_image(image_path):
    """Load and preprocess a single image for prediction"""
    try:
        img = Image.open(image_path)
        # Keep a copy of the original image without any preprocessing
        original_img = img.copy()
        
        # Process for model input
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((16, 16))
        img_array = np.array(img).flatten() / 255.0  # Normalize here
        return img_array, original_img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

def encrypt_data(context, data):
    """Encrypt the input data using the provided context"""
    encrypted_data = []
    for sample in data:
        scaled_sample = sample.astype(np.float64) * 0.1  # Changed from sample / 255.0
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
    def __init__(self, learning_rate=0.01, num_iterations=50):  # Changed default from 0.001 to 0.01
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.context = None
        self.training_history = {'loss': [], 'accuracy': []}
        self.reg_lambda = 0.01  # L2 regularization parameter

    def safe_sigmoid_approx(self, x):
        """Better sigmoid approximation using polynomial approximation"""
        try:
            half = ts.ckks_vector(self.context, [0.5])
            # More accurate approximation with cubic terms
            coef1 = ts.ckks_vector(self.context, [0.25])  # Increased from 0.15
            return half + x * coef1 - x * x * x * ts.ckks_vector(self.context, [0.01])
        except Exception as e:
            print(f"Error in sigmoid approximation: {e}")
            raise

    def calculate_metrics(self, X_encrypted, y):
        """Calculate loss and accuracy for current iteration"""
        predictions, raw_predictions = self.predict(X_encrypted)
        accuracy = np.mean(predictions == y) * 100
        epsilon = 1e-15
        raw_predictions = np.clip(raw_predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(raw_predictions) + (1 - y) * np.log(1- raw_predictions))
        # Add regularization term to loss
        l2_loss = 0.5 * self.reg_lambda * np.sum(self.weights**2) / len(y)
        return loss + l2_loss, accuracy

    def fit(self, X_encrypted, y, context, progress_callback=None):
        """Train the model using encrypted data"""
        try:
            self.context = context
            num_samples = len(X_encrypted)
            num_features = len(X_encrypted[0].decrypt())
            
            # Initialize with zeros instead of random small weights
            self.weights = np.zeros(num_features)
            self.bias = 0.0
            
            # Adaptive batch size based on dataset size
            batch_size = min(32, max(1, num_samples // 10))
            
            for iteration in range(self.num_iterations):
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
                
                # Update weights and bias with regularization
                weight_gradients /= num_samples
                weight_gradients += self.reg_lambda * self.weights / num_samples  # Add regularization
                bias_gradient /= num_samples
                self.weights -= self.learning_rate * weight_gradients
                self.bias -= self.learning_rate * bias_gradient
                
                # Calculate and store metrics
                loss, accuracy = self.calculate_metrics(X_encrypted, y)
                self.training_history['loss'].append(loss)
                self.training_history['accuracy'].append(accuracy)
                
                if progress_callback:
                    progress = ((iteration + 1) / self.num_iterations) * 100
                    progress_callback(
                        iteration + 1,
                        progress,
                        f"Iteration {iteration + 1}/{self.num_iterations}",
                        {"accuracy": accuracy, "loss": loss}
                    )

        except Exception as e:
            print(f"Error in training: {e}")
            raise

    def predict(self, X_encrypted):
        """Make predictions on encrypted data"""
        if self.weights is None or self.context is None:
            raise ValueError("Model not trained yet")
        
        predictions = []
        raw_predictions = []
        weights_encrypted = ts.ckks_vector(self.context, self.weights.tolist())
        
        for x_enc in X_encrypted:
            try:
                z = x_enc.dot(weights_encrypted)
                pred_encrypted = self.safe_sigmoid_approx(z)
                pred_value = float(pred_encrypted.decrypt()[0])
                raw_predictions.append(pred_value)
                predictions.append(1 if pred_value > 0.5 else 0)
            except Exception as e:
                print(f"Prediction error: {e}")
                predictions.append(0)
                raw_predictions.append(0.0)
        
        return np.array(predictions), np.array(raw_predictions)

def send_progress(message, progress=None, type="info", metrics=None):
    """Send progress updates to the queue"""
    progress_queue.put({
        "message": message,
        "progress": progress,
        "type": type,
        "metrics": metrics
    })

def train_model_thread(use_full_dataset=True, max_samples=100, learning_rate=0.01, num_iterations=50):
    """Training function that runs in a separate thread"""
    try:
        global model, context

        # Load dataset
        positive_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Positive"
        negative_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Negative"

        send_progress("Loading and processing images...")
        X_pos, y_pos = load_and_flatten_images(positive_path, 1)
        X_neg, y_neg = load_and_flatten_images(negative_path, 0)

        send_progress(f"Raw dataset: {len(X_pos)} positive, {len(X_neg)} negative")
        
        # Balance the dataset
        pos_count = len(X_pos)
        neg_count = len(X_neg)
        min_count = min(pos_count, neg_count)
        
        if pos_count > neg_count:
            indices = np.random.choice(pos_count, min_count, replace=False)
            X_pos = [X_pos[i] for i in indices]
            y_pos = [y_pos[i] for i in indices]
        elif neg_count > pos_count:
            indices = np.random.choice(neg_count, min_count, replace=False)
            X_neg = [X_neg[i] for i in indices]
            y_neg = [y_neg[i] for i in indices]
        
        send_progress(f"Balanced dataset: {min_count} positive, {min_count} negative")

        X = np.array(X_pos + X_neg)
        y = np.array(y_pos + y_neg)

        send_progress(f"Total samples: {len(X)}")

        if not use_full_dataset:
            sss = StratifiedKFold(n_splits=len(X)//max_samples, shuffle=True, random_state=42)
            train_idx, _ = next(sss.split(X, y))
            X = X[train_idx[:max_samples]]
            y = y[train_idx[:max_samples]]
            send_progress(f"Using {len(X)} stratified samples")

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        send_progress("Creating encryption context...")
        context = create_ckks_context()

        send_progress("Encrypting training data...")
        start_time = time.time()
        X_train_enc = encrypt_data(context, X_train)
        X_test_enc = encrypt_data(context, X_test)
        send_progress(f"Encryption completed in {time.time() - start_time:.2f} seconds")

        send_progress(f"Initializing model (lr={learning_rate}, iterations={num_iterations})...")
        model = EncryptedLogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations)

        def progress_callback(iteration, progress, message, metrics):
            send_progress(message, progress=progress, metrics=metrics)

        send_progress("Starting training...")
        start_time = time.time()
        model.fit(X_train_enc, y_train, context, progress_callback)
        training_time = time.time() - start_time

        # Evaluate on test set
        send_progress("Evaluating model...")
        y_pred, _ = model.predict(X_test_enc)
        test_accuracy = calculate_accuracy(y_test, y_pred)

        # Calculate metrics
        TP = sum((y_test == 1) & (y_pred == 1))
        TN = sum((y_test == 0) & (y_pred == 0))
        FP = sum((y_test == 0) & (y_pred == 1))
        FN = sum((y_test == 1) & (y_pred == 0))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Add confusion matrix information
        send_progress(f"Confusion Matrix:\n"
                     f"True Positive: {TP}, False Positive: {FP}\n"
                     f"True Negative: {TN}, False Negative: {FN}\n"
                     f"Positive Samples: {sum(y_test == 1)}, Negative Samples: {sum(y_test == 0)}")

        final_message = (
            f"Training complete!\n"
            f"Training time: {training_time:.2f} seconds\n"
            f"Test accuracy: {test_accuracy:.2f}%\n"
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        send_progress(final_message, type="success")

    except Exception as e:
        send_progress(f"Error during training: {str(e)}", type="error")

def predict_image(file_path, filename="Unknown file"):
    """Make predictions on an image"""
    try:
        # Load and preprocess the image
        img_array, original_img = load_single_image(file_path)
        if img_array is None:
            return {"error": "Error processing image"}
        
        # Start timing for encryption
        encryption_start = time.time()
        # Encrypt the image
        img_enc = encrypt_data(context, [img_array])[0]
        encryption_time = time.time() - encryption_start
        
        # Start timing for prediction and decryption
        decrypt_start = time.time()
        # Make prediction
        predictions, raw_predictions = model.predict([img_enc])
        prediction = predictions[0]
        confidence = raw_predictions[0] if prediction == 1 else 1 - raw_predictions[0]
        decryption_time = time.time() - decrypt_start
        
        # Use Figure instead of pyplot to avoid threading issues
        fig = Figure(figsize=(12, 5))
        
        # Original image
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(original_img, cmap='gray' if original_img.mode == 'L' else None)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Processed image with prediction
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img_array.reshape(16, 16), cmap='gray')
        result = "Positive (Sickle Cell)" if prediction == 1 else "Negative (Healthy)"
        ax2.set_title(f"Prediction: {result}\nConfidence: {confidence:.2f}")
        ax2.axis('off')
        
        # Save figure to memory buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        return {
            "prediction": result,
            "confidence": float(confidence),
            "image": img_str,
            "filename": filename,
            "encryption_time": f"{encryption_time:.3f}",
            "decryption_time": f"{decryption_time:.3f}",
            "total_processing_time": f"{(encryption_time + decryption_time):.3f}"
        }
        
    except Exception as e:
        return {"error": str(e)}
    if __name__ == "__main__":
        send_progress("Starting the training process...")
        train_thread = Thread(target=train_model_thread, kwargs={"use_full_dataset": True, "max_samples": 100})
        train_thread.start()

        while train_thread.is_alive():
            while not progress_queue.empty():
                progress = progress_queue.get()
                print(progress)  # Print progress messages
