import tenseal as ts
import numpy as np

def create_ckks_context():
    """Create the CKKS context with optimized parameters"""
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
        print(f"Error creating CKKS context: {e}")
        raise

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
