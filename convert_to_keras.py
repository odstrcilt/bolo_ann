"""
Script to convert the custom boloNN network format to a standard Keras MLP model.
The network architecture combines a linear branch with an MLP branch.
"""

import h5py
import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers, Model
except ImportError:
    try:
        import keras
        from keras import layers, Model
    except ImportError:
        raise ImportError("Please install tensorflow or keras to use this script")


def load_network(filepath):
    """Loads the network parameters from an HDF5 file into NumPy arrays."""
    params = []
    with h5py.File(filepath, 'r') as f:
        # Load MLP parameters
        i = 0
        while f'mlp/layer_{i}/W' in f:
            W = f[f'mlp/layer_{i}/W'][:]
            b = f[f'mlp/layer_{i}/b'][:]
            params.append([W, b])
            i += 1

        # Load linear part parameters
        Wlin = f['linear/Wlin'][:]
        W0 = f['linear/W0'][:]

    print(f"Network loaded successfully from {filepath}")
    print(f"  - MLP layers: {len(params)}")
    print(f"  - Input dimension: {Wlin.shape[0]}")
    print(f"  - Output dimension: {W0.shape[0]}")
    return params, Wlin, W0


def create_keras_model(params, Wlin, W0):
    """
    Creates a Keras model that replicates the custom network architecture.
    
    The network combines:
    - Linear branch: W0 + x @ Wlin
    - MLP branch: MLP(x) with SiLU activations
    - Final output: linear_branch + mlp_branch
    """
    input_dim = Wlin.shape[0]
    output_dim = W0.shape[0]
    
    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Linear branch: W0 + x @ Wlin
    linear_branch = layers.Dense(
        output_dim,
        use_bias=True,
        name='linear_branch'
    )(inputs)
    
    # MLP branch
    x = inputs
    for i, (W, b) in enumerate(params[:-1]):
        x = layers.Dense(
            W.shape[1],
            activation='silu',
            name=f'mlp_layer_{i}'
        )(x)
    
    # Last MLP layer (no activation)
    W_last, b_last = params[-1]
    mlp_branch = layers.Dense(
        W_last.shape[1],
        use_bias=True,
        activation=None,
        name=f'mlp_layer_{len(params)-1}'
    )(x)
    
    # Combine linear and MLP branches
    outputs = layers.Add(name='output')([linear_branch, mlp_branch])
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='boloNN_keras_model')
    
    # Set weights
    # Keras Dense layers store weights as (input_dim, output_dim) in the kernel
    # Linear branch: weights=Wlin, bias=W0
    model.get_layer('linear_branch').set_weights([Wlin, W0])
    
    # MLP layers
    for i, (W, b) in enumerate(params):
        layer_name = f'mlp_layer_{i}'
        model.get_layer(layer_name).set_weights([W, b])
    
    return model


def convert_to_keras(input_h5_path, output_h5_path=None):
    """
    Main conversion function.
    
    Args:
        input_h5_path: Path to the custom HDF5 network file
        output_h5_path: Path to save the Keras model (default: input_h5_path with '_keras.h5' suffix)
    """
    # Load network parameters
    params, Wlin, W0 = load_network(input_h5_path)
    
    # Create Keras model
    print("\nCreating Keras model...")
    model = create_keras_model(params, Wlin, W0)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Save model
    if output_h5_path is None:
        output_h5_path = input_h5_path.replace('.h5', '_keras.h5')
    
    print(f"\nSaving Keras model to {output_h5_path}...")
    model.save(output_h5_path)
    print(f"✓ Keras model saved successfully!")
    
    return model


if __name__ == '__main__':
    import sys
    
    input_file = 'trained_network.h5'
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    model = convert_to_keras(input_file, output_file)

    # Save the keras model
    if output_file is None:
        output_file = input_file.replace('.h5', '_keras.h5')
    print(f"\nSaving Keras model to {output_file} (again)...")
    model.save(output_file)
    print(f"✓ Keras model saved successfully (again)!")

    # Optional: Test the model with a dummy input
    print("\nTesting model with dummy input...")
    input_dim = model.input_shape[1]
    dummy_input = np.random.randn(1, input_dim).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0, :5]}")

