import numpy as np
import tenseal as ts

class EncryptedLogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=50):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def safe_sigmoid_approx(self, x):
        """Sigmoid approximation using a linear function"""
        try:
            half = ts.ckks_vector(self.context, [0.5])
            linear = ts.ckks_vector(self.context, [0.15])  # Approximation slope
            return half + x * linear
        except Exception as e:
            print(f"Error in sigmoid approximation: {e}")
            raise

    def fit(self, X_encrypted, y, context):
        """Train the model using encrypted data"""
        try:
            self.context = context
            num_samples = len(X_encrypted)
            num_features = len(X_encrypted[0].decrypt())  # Decrypt to get size
            
            self.weights = np.random.randn(num_features) * 0.01
            self.bias = 0.0
            
            batch_size = 10  # Process in batches
            
            for iteration in range(self.num_iterations):
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
                
                weights_encrypted = ts.ckks_vector(context, self.weights.tolist())
                weight_gradients = np.zeros(num_features)
                bias_gradient = 0.0

                for i in range(0, num_samples, batch_size):
                    batch_gradients = np.zeros(num_features)
                    batch_bias = 0.0

                    for j in range(i, min(i + batch_size, num_samples)):
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
                
                weight_gradients /= num_samples
                bias_gradient /= num_samples
                
                self.weights -= self.learning_rate * weight_gradients
                self.bias -= self.learning_rate * bias_gradient
                
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
                predictions.append(0)  # Default to negative prediction

        return np.array(predictions)
