import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
import os

def create_model_architecture():
    """Recreate the model architecture based on the notebook"""
    model = Sequential()
    
    # Input layer (explicitly defined to avoid batch_shape issues)
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(256,256,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def try_load_weights_manually(model, model_path):
    """Try to load weights manually from the model file"""
    try:
        # Try to load just the weights
        model.load_weights(model_path, by_name=True)
        print(f"✅ Successfully loaded weights from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to load weights from {model_path}: {e}")
        return False

def create_compatible_model():
    """Create a new compatible model and save it"""
    print("🔧 Creating new compatible model architecture...")
    
    # Create the model with current TensorFlow version
    model = create_model_architecture()
    
    print("📋 Model architecture created successfully!")
    print("📊 Model summary:")
    model.summary()
    
    # Save the new model in both formats
    tf_version = tf.__version__.replace('.', '_')
    
    print(f"\n💾 Saving compatible model files...")
    
    # Save with version-specific names
    h5_filename = f"GLD_Binary_Classification_Final_TF{tf_version}.h5"
    keras_filename = f"GLD_Binary_Classification_Final_TF{tf_version}.keras"
    
    model.save(h5_filename)
    print(f"✅ Saved: {h5_filename}")
    
    model.save(keras_filename)
    print(f"✅ Saved: {keras_filename}")
    
    # Save with original names for app compatibility
    model.save("GLD_Binary_Classification_Final.h5")
    print(f"✅ Saved: GLD_Binary_Classification_Final.h5")
    
    model.save("GLD_Binary_Classification_Final.keras")
    print(f"✅ Saved: GLD_Binary_Classification_Final.keras")
    
    print(f"\n🎉 Created new compatible model with TensorFlow {tf.__version__}!")
    print("Note: This is a fresh model without trained weights.")
    print("You'll need to either:")
    print("1. Re-train the model with your dataset")
    print("2. Or use the original model files if you can find a compatible TensorFlow version")
    
    return model

def main():
    print("🔄 Advanced Model Compatibility Fix Tool")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"🔍 Current TensorFlow version: {tf.__version__}")
    
    # Check for existing model files
    model_files = [
        "GLD_Binary_Classification_Final.h5",
        "GLD_Binary_Classification_Final.keras"
    ]
    
    existing_files = [f for f in model_files if os.path.exists(f)]
    
    if existing_files:
        print(f"📁 Found existing model files: {existing_files}")
        
        # Try to create new model and load weights
        model = create_model_architecture()
        
        # Try to load weights from existing files
        weights_loaded = False
        for model_file in existing_files:
            if try_load_weights_manually(model, model_file):
                weights_loaded = True
                break
        
        if weights_loaded:
            print("\n💾 Saving model with loaded weights...")
            tf_version = tf.__version__.replace('.', '_')
            
            # Save in multiple formats
            model.save(f"GLD_Binary_Classification_Final_TF{tf_version}.h5")
            model.save(f"GLD_Binary_Classification_Final_TF{tf_version}.keras")
            model.save("GLD_Binary_Classification_Final.h5")
            model.save("GLD_Binary_Classification_Final.keras")
            
            print("✅ Successfully created compatible model with loaded weights!")
            print("You can now run: streamlit run app.py")
            return True
        else:
            print("\n⚠️ Could not load weights from existing files.")
            print("Creating new model architecture without trained weights...")
    else:
        print("📁 No existing model files found.")
        print("Creating new model architecture...")
    
    # Create new compatible model
    create_compatible_model()
    
    print("\n📋 Next steps:")
    print("1. If you have the original training data, re-train the model")
    print("2. Or try running the app with the new model (will need training)")
    print("3. Or find a compatible TensorFlow version for the original model")
    
    return True

if __name__ == "__main__":
    main() 