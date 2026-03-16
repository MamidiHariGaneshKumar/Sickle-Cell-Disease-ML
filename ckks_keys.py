import tenseal as ts

def setup_ckks_encryption():
    # Create CKKS context with parameters suitable for your problem
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 60])
    context.generate_galois_keys()  # Generate necessary Galois keys (not secret/public directly)
    context.generate_relin_keys()  # Generate relinearization keys
    return context

# Generate keys and initialize the context
context = setup_ckks_encryption()
print("Keys generated.")

