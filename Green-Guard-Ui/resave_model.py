import tensorflow as tf
import os

def check_tensorflow_version():
    """Check and display TensorFlow version"""
    version = tf.__version__
    print(f"Current TensorFlow version: {version}")
    return version

def resave_model_with_compatibility():
    """Re-save model with current TensorFlow version for compatibility"""
    
    print(" Checking TensorFlow version...")
    tf_version = check_tensorflow_version()
    
    # Define model files to try
    model_files = [
        "GLD_Binary_Classification_Final.h5",
        "GLD_Binary_Classification_Final.keras"
    ]
    
    # Check which files exist
    existing_files = [f for f in model_files if os.path.exists(f)]
    
    if not existing_files:
        print(" Error: No model files found!")
        print("Expected files:")
        for file in model_files:
            print(f"  - {file}")
        return False
    
    print(f" Found model files: {existing_files}")
    
    # Try to load and re-save each existing file
    for model_file in existing_files:
        try:
            print(f"\n🔄 Attempting to load {model_file}...")
            
            # Load the model
            model = tf.keras.models.load_model(model_file)
            print(f" Successfully loaded {model_file}")
            
            # Create new filenames with current TF version
            tf_version_clean = tf_version.replace('.', '_')
            
            # Save in both formats for maximum compatibility
            h5_filename = f"GLD_Binary_Classification_Final_TF{tf_version_clean}.h5"
            keras_filename = f"GLD_Binary_Classification_Final_TF{tf_version_clean}.keras"
            
            print(f" Saving model as {h5_filename}...")
            model.save(h5_filename)
            print(f" Successfully saved {h5_filename}")
            
            print(f" Saving model as {keras_filename}...")
            model.save(keras_filename)
            print(f" Successfully saved {keras_filename}")
            
            # Also save with original names for app compatibility
            print(f" Saving model as GLD_Binary_Classification_Final.h5...")
            model.save("GLD_Binary_Classification_Final.h5")
            print(f" Successfully saved GLD_Binary_Classification_Final.h5")
            
            print(f" Saving model as GLD_Binary_Classification_Final.keras...")
            model.save("GLD_Binary_Classification_Final.keras")
            print(f" Successfully saved GLD_Binary_Classification_Final.keras")
            
            print(f"\n🎉 Model successfully re-saved with TensorFlow {tf_version}!")
            print("You can now run the Streamlit app:")
            print("  streamlit run app.py")
            
            return True
            
        except Exception as e:
            print(f" Failed to process {model_file}: {e}")
            continue
    
    print("\n All model files failed to load!")
    print("This might be due to:")
    print("1. Corrupted model files")
    print("2. Incompatible model architecture")
    print("3. Missing dependencies")
    print("\nTry:")
    print("1. Re-training the model with current TensorFlow version")
    print("2. Using a compatible TensorFlow version")
    print("3. Checking if all required packages are installed")
    
    return False

if __name__ == "__main__":
    print(" Model Compatibility Fix Tool")
    print("=" * 40)
    
    success = resave_model_with_compatibility()
    
    if success:
        print("\n Model compatibility issue resolved!")
    else:
        print("\n Could not resolve model compatibility issue.")
        print("Please check the error messages above for guidance.")
