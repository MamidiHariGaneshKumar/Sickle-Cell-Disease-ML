from flask import Flask, render_template_string, jsonify, request, Response
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


app = Flask(__name__)

# Global variables to store model and context
model = None
context = None
progress_queue = Queue()

# HTML template with enhanced styling and features
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enhanced Sickle Cell Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .control-panel {
            margin-bottom: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        .progress-container {
            margin: 20px 0;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            width: 0%;
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease;
        }
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }
        .metric-card {
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
        }
        .log-entry {
            margin: 5px 0;
            padding: 5px;
            border-radius: 4px;
        }
        .log-info { background-color: #e3f2fd; }
        .log-success { background-color: #e8f5e9; }
        .log-error { background-color: #ffebee; }
        .prediction-container {
            margin-top: 20px;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .prediction-image {
            max-width: 100%;
            margin-top: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #1976D2;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        .hidden {
            display: none;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .setting-item {
            padding: 10px;
            background-color: #fff;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .result-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
        }
        .result-card img {
            max-width: 100%;
            border-radius: 4px;
        }
        .result-card h4 {
            margin-top: 10px;
            margin-bottom: 5px;
        }
        .result-info {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sickle Cell Detection Model</h1>
        
        <div class="control-panel">
            <h2>Training Configuration</h2>
            <div class="settings-grid">
                <div class="setting-item">
                    <input type="checkbox" id="useFullDataset" checked>
                    <label for="useFullDataset">Use Full Dataset</label>
                </div>
                <div class="setting-item">
                    <label for="maxSamples">Max Samples:</label>
                    <input type="number" id="maxSamples" value="100" min="10" max="1000" disabled>
                </div>
                <div class="setting-item">
                    <label for="learningRate">Learning Rate:</label>
                    <input type="number" id="learningRate" value="0.001" step="0.0001" min="0.0001" max="0.01">
                </div>
                <div class="setting-item">
                    <label for="numIterations">Number of Iterations:</label>
                    <input type="number" id="numIterations" value="50" min="10" max="200">
                </div>
            </div>
            <button id="startTraining">Start Training</button>
        </div>
        
        <div class="progress-container">
            <h2>Training Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar"></div>
            </div>
            <div class="metrics-container">
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <div class="metric-value" id="accuracyValue">-</div>
                </div>
                <div class="metric-card">
                    <h3>Loss</h3>
                    <div class="metric-value" id="lossValue">-</div>
                </div>
                <div class="metric-card">
                    <h3>Time Elapsed</h3>
                    <div class="metric-value" id="timeValue">-</div>
                </div>
            </div>
            <div class="log-container" id="logContainer"></div>
        </div>
        
        <div class="prediction-container hidden" id="predictionSection">
            <h2>Make Predictions</h2>
            <input type="file" id="imageUpload" accept="image/*">
            <button id="predict" disabled>Predict</button>
            
            <div id="timingMetrics" class="metrics-container hidden">
                <div class="metric-card">
                    <h3>Encryption Time</h3>
                    <div class="metric-value" id="encryptionTime">-</div>
                    <div class="metric-label">seconds</div>
                </div>
                <div class="metric-card">
                    <h3>Decryption Time</h3>
                    <div class="metric-value" id="decryptionTime">-</div>
                    <div class="metric-label">seconds</div>
                </div>
                <div class="metric-card">
                    <h3>Total Processing Time</h3>
                    <div class="metric-value" id="totalTime">-</div>
                    <div class="metric-label">seconds</div>
                </div>
            </div>
            
            <div id="resultsList" class="results-grid"></div>
        </div>
    </div>

    <script>
        let trainingInProgress = false;
        let eventSource = null;
        let startTime = null;
        let resultCounter = 0;
        
        function updateTimer() {
            if (startTime && trainingInProgress) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                document.getElementById('timeValue').textContent = 
                    `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
        }
        
        document.getElementById('useFullDataset').addEventListener('change', function(e) {
            document.getElementById('maxSamples').disabled = e.target.checked;
        });
        
        document.getElementById('startTraining').addEventListener('click', function() {
            if (trainingInProgress) return;
            
            trainingInProgress = true;
            startTime = Date.now();
            setInterval(updateTimer, 1000);
            
            this.disabled = true;
            document.getElementById('logContainer').innerHTML = '';
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('predictionSection').classList.add('hidden');
            document.getElementById('accuracyValue').textContent = '-';
            document.getElementById('lossValue').textContent = '-';
            document.getElementById('timeValue').textContent = '0:00';
            document.getElementById('resultsList').innerHTML = '';
            resultCounter = 0;
            
            if (eventSource) {
                eventSource.close();
            }
            
            eventSource = new EventSource('/progress');
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.progress !== null) {
                    document.getElementById('progressBar').style.width = `${data.progress}%`;
                }
                
                if (data.metrics) {
                    document.getElementById('accuracyValue').textContent = `${data.metrics.accuracy.toFixed(2)}%`;
                    document.getElementById('lossValue').textContent = data.metrics.loss.toFixed(4);
                }
                
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${data.type || 'info'}`;
                logEntry.textContent = data.message;
                document.getElementById('logContainer').appendChild(logEntry);
                logEntry.scrollIntoView({ behavior: 'smooth' });
                
                if (data.type === 'success') {
                    trainingInProgress = false;
                    document.getElementById('startTraining').disabled = false;
                    document.getElementById('predictionSection').classList.remove('hidden');
                    document.getElementById('predict').disabled = false;
                    eventSource.close();
                }
            };
            
            fetch('/start_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    useFullDataset: document.getElementById('useFullDataset').checked,
                    maxSamples: parseInt(document.getElementById('maxSamples').value),
                    learningRate: parseFloat(document.getElementById('learningRate').value),
                    numIterations: parseInt(document.getElementById('numIterations').value)
                })
            });
        });
        
        document.getElementById('imageUpload').addEventListener('change', function() {
            if (this.files.length > 0) {
                document.getElementById('predict').disabled = false;
            } else {
                document.getElementById('predict').disabled = true;
            }
        });
        
        document.getElementById('predict').addEventListener('click', function() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select an image first');
        return;
    }
    
    // Show a loading indicator
    document.getElementById('predict').disabled = true;
    document.getElementById('predict').textContent = 'Processing...';
    
    // Clear previous results when starting a new prediction
    document.getElementById('resultsList').innerHTML = '';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('filename', file.name); // Send the original filename
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Enable the predict button again
        document.getElementById('predict').disabled = false;
        document.getElementById('predict').textContent = 'Predict';
        document.getElementById('timingMetrics').classList.remove('hidden');
        
        // Update timing metrics
        document.getElementById('encryptionTime').textContent = data.encryption_time;
        document.getElementById('decryptionTime').textContent = data.decryption_time;
        document.getElementById('totalTime').textContent = data.total_processing_time;
        
        // Create the result card for the current image
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        resultCard.innerHTML = `
            <h3>Result: ${data.filename}</h3>
            <img src="data:image/png;base64,${data.image}" alt="Prediction Result">
            <h4>Classification: ${data.prediction}</h4>
            <div class="result-info">
                <span>Confidence: ${(data.confidence * 100).toFixed(2)}%</span>
                <span>Processing Time: ${data.total_processing_time}s</span>
            </div>
        `;
        
        // Add to the results list
        const resultsList = document.getElementById('resultsList');
        resultsList.appendChild(resultCard);
        
        // Simple way to reset the file input
        fileInput.value = '';
        document.getElementById('predict').disabled = true;
    })
    .catch(error => {
        alert('Error making prediction: ' + error.message);
        document.getElementById('predict').disabled = false;
        document.getElementById('predict').textContent = 'Predict';
    });
});
    </script>
</body>
</html>
'''

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
        # Keep a copy of the original image without any preprocessing
        original_img = img.copy()
        
        # Process for model input
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((16, 16))
        img_array = np.array(img).flatten()
        return img_array, original_img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

def encrypt_data(context, data):
    """Encrypt the input data using the provided context"""
    encrypted_data = []
    for sample in data:
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
        accuracy = np.mean(predictions == y) * 100
        epsilon = 1e-15
        raw_predictions = np.clip(raw_predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(raw_predictions) + (1 - y) * np.log(1- raw_predictions))
        return loss, accuracy

    def fit(self, X_encrypted, y, context, progress_callback=None):
        """Train the model using encrypted data"""
        try:
            self.context = context
            num_samples = len(X_encrypted)
            num_features = len(X_encrypted[0].decrypt())
            
            # Initialize with small weights
            self.weights = np.random.randn(num_features) * 0.01
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
                
                # Update weights and bias
                weight_gradients /= num_samples
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

def train_model_thread(use_full_dataset=True, max_samples=100, learning_rate=0.001, num_iterations=50):
    """Training function that runs in a separate thread"""
    try:
        global model, context

        # Load dataset
        positive_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Positive"
        negative_path = "C:/Users/mamid/OneDrive/Desktop/Sickle Cell Data set/Negative"

        send_progress("Loading and processing images...")
        X_pos, y_pos = load_and_flatten_images(positive_path, 1)
        X_neg, y_neg = load_and_flatten_images(negative_path, 0)

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

        final_message = (
            f"Training complete!\n"
            f"Training time: {training_time:.2f} seconds\n"
            f"Test accuracy: {test_accuracy:.2f}%\n"
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        send_progress(final_message, type="success")

    except Exception as e:
        send_progress(f"Error during training: {str(e)}", type="error")

@app.route('/')
def home():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start the training process"""
    data = request.get_json()
    use_full_dataset = data.get('useFullDataset', True)
    max_samples = data.get('maxSamples', 100)
    learning_rate = data.get('learningRate', 0.001)
    num_iterations = data.get('numIterations', 50)
    
    Thread(target=train_model_thread, args=(use_full_dataset, max_samples, learning_rate, num_iterations)).start()
    return jsonify({"status": "started"})

@app.route('/progress')
def progress():
    """Stream training progress updates"""
    def generate():
        while True:
            progress_data = progress_queue.get()
            yield f"data: {json.dumps(progress_data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# Replace the matplotlib part of your predict route with this:
from matplotlib.figure import Figure
from io import BytesIO
import base64

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on uploaded images with decryption time tracking"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    filename = request.form.get('filename', 'Unknown file')
    
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = "temp_images"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Generate a unique filename
        timestamp = int(time.time() * 1000)
        temp_path = os.path.join(temp_dir, f"img_{timestamp}.jpg")
        file.save(temp_path)
        
        # Load and preprocess the image
        img_array, original_img = load_single_image(temp_path)
        if img_array is None:
            return jsonify({"error": "Error processing image"})
        
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
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            "prediction": result,
            "confidence": float(confidence),
            "image": img_str,
            "filename": filename,
            "encryption_time": f"{encryption_time:.3f}",
            "decryption_time": f"{decryption_time:.3f}",
            "total_processing_time": f"{(encryption_time + decryption_time):.3f}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
    