import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import base64
import logging

st.set_page_config(
    page_title="GreenGuard - Guava Leaf Classifier",
    page_icon="🌿",
    layout="wide"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Additional detailed logging for debugging
logger.info(f"TensorFlow version: {__import__('tensorflow').__version__}")
logger.info(f"Loading model from file: GLD_Binary_Classification_Final.keras")

def set_bg_with_theme(light_img, dark_img, dark_mode):
    # Use image file names directly without prefix
    img_file = dark_img if dark_mode else light_img
    with open(img_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string.decode()}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        transition: background-image 1s ease-in-out;
        font-family: 'Segoe UI', sans-serif;
    }}
    .title {{
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: {"#FFFFFF" if dark_mode else "#1B4332"};
        text-shadow: 2px 2px 8px {"#000000" if dark_mode else "#FFFFFF"};
    }}
    .subheader {{
        text-align: center;
        font-size: 1.2em;
        color: {"#EEEEEE" if dark_mode else "#1B4332"};
        font-weight: bold;
        text-shadow: 1px 1px 4px {"#000000" if dark_mode else "#FFFFFF"};
    }}
    .upload-box {{
        background: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 15px;
        margin: 10px auto;
        max-width: 90%;
        width: 450px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        color: black;
        font-weight: bold;
    }}
    .result-card {{
        background: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: {"#000000" if dark_mode else "#000000"};
        font-weight: bold;
        transition: transform 0.3s ease;
        max-width: 100%;
        word-wrap: break-word;
    }}
    .result-card:hover {{
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }}
    .image-container {{
        text-align: center;
        margin: 10px 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    .image-container img {{
        max-width: 100%;
        height: auto;
        border-radius: 8px;
    }}
    .summary {{
        background: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        max-width: 90%;
        width: 400px;
        margin: 30px auto;
        font-size: 18px;
        font-weight: bold;
        color: {"#000000" if dark_mode else "#000000"};
    }}

    /* Responsive adjustments */
    @media (max-width: 600px) {{
        .title {{
            font-size: 2em;
        }}
        .subheader {{
            font-size: 1em;
        }}
        .upload-box {{
            width: 95%;
            padding: 10px;
            font-size: 1em;
        }}
        .result-card {{
            font-size: 0.9em;
            padding: 8px;
        }}
        .summary {{
            width: 95%;
            font-size: 1em;
            padding: 15px;
        }}
        /* Make columns wrap on small screens */
        .stApp > div[data-testid="stHorizontalBlock"] > div {{
            flex-wrap: wrap;
            justify-content: center;
        }}
        /* Force Streamlit columns to stack vertically on mobile */
        .stApp .css-1lcbmhc.e1fqkh3o3 {{
            flex-direction: column !important;
            align-items: center !important;
        }}
        
        /* Improve image display */
        .stImage > img {{
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
            transition: transform 0.3s ease !important;
        }}
        .stImage > img:hover {{
            transform: scale(1.02) !important;
        }}
        
        /* Ensure consistent image sizing */
        .stImage {{
            text-align: center !important;
            margin: 10px 0 !important;
        }}
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Maintain files state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'clear_files' not in st.session_state:
    st.session_state.clear_files = False

dark_mode = st.toggle("🌙 Dark Mode", value=False)
set_bg_with_theme("Background_img_2.jpg", "Background_dark_img_2.jpg", dark_mode)

def preprocess_image_for_display(img, target_size=(256, 256)):
    """Preprocess image for consistent display and model input"""
    # Resize image for model input
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    return img_resized

@st.cache_resource
def load_cnn_model():
    # Try different model formats in order of preference
    model_paths = [
        "GLD_Binary_Classification_Final.keras",
        "GLD_Binary_Classification_Final.h5"
    ]
    
    for model_path in model_paths:
        try:
            logger.info(f"Attempting to load model from {model_path}...")
            model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}.")
            return model
        except Exception as e:
            logger.warning(f"Failed to load {model_path}: {e}")
            continue
    
    # If all attempts fail, show error and provide solution
    logger.error("All model loading attempts failed")
    st.error("Error loading model: All model files failed to load due to version incompatibility.")
    
    # Provide detailed instructions for fixing the issue
    st.markdown("""
    <div style='background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 10px 0;'>
    <h4 style='color: #856404; margin-top: 0;'>🔧 Model Loading Issue - Solution Required</h4>
    <p style='color: #856404; margin-bottom: 10px;'>
    The model files could not be loaded due to TensorFlow/Keras version incompatibility. 
    This happens when models are saved with a different version than what's currently installed.
    </p>
    
    <h5 style='color: #856404;'>📋 Steps to Fix:</h5>
    <ol style='color: #856404;'>
    <li><strong>Check your TensorFlow version:</strong><br>
    <code>python -c "import tensorflow as tf; print(tf.__version__)"</code></li>
    
    <li><strong>Re-save the model using the current version:</strong><br>
    <pre style='background-color: #f8f9fa; padding: 10px; border-radius: 3px;'>
import tensorflow as tf

# Try loading the .h5 file first
try:
    model = tf.keras.models.load_model('GLD_Binary_Classification_Final.h5')
    model.save('GLD_Binary_Classification_Final.keras')
    print("Model re-saved successfully!")
except Exception as e:
    print(f"Error: {e}")
    
# If that fails, try the .keras file
try:
    model = tf.keras.models.load_model('GLD_Binary_Classification_Final.keras')
    model.save('GLD_Binary_Classification_Final.h5')
    print("Model re-saved successfully!")
except Exception as e:
    print(f"Error: {e}")
    </pre></li>
    
    <li><strong>Restart the Streamlit app:</strong><br>
    <code>streamlit run app.py</code></li>
    </ol>
    
    <p style='color: #856404; font-size: 0.9em; margin-top: 15px;'>
    <strong>Note:</strong> If you continue to have issues, you may need to re-train the model 
    using your current TensorFlow version or use a compatible TensorFlow version.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    return None

model = load_cnn_model()

if model is not None:
    st.markdown('<div class="title">🌿 GreenGuard 🌿</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">AI-powered Guava Leaf Disease Detector</div>', unsafe_allow_html=True)

    st.markdown('<div class="upload-box">📤 Upload Guava Leaf Images (JPG/PNG):</div>', unsafe_allow_html=True)

    # Reset file_uploader after clear
    key = "file_uploader"
    if st.session_state.clear_files:
        key = "file_uploader_new"  # generate new key to force re-render
        st.session_state.clear_files = False

    uploaded_files = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True, 
        label_visibility="collapsed",
        key=key
    )

    # Save uploaded files in session_state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    if st.session_state.uploaded_files:
        analyze_btn = st.button("🔍 Analyze All Uploaded Leaves", key="analyze_btn")
        clear_btn = st.button("❌ Clear Files", key="clear_btn")

        if clear_btn:
            st.session_state.uploaded_files = []
            st.session_state.clear_files = True  # trigger re-render
            st.experimental_rerun()

        if analyze_btn:
            healthy_count = 0
            diseased_count = 0

            cols = st.columns(len(st.session_state.uploaded_files))

            for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
                img = Image.open(uploaded_file)
                img_resized = preprocess_image_for_display(img, target_size=(256, 256))
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner(f'Analyzing {uploaded_file.name}...'):
                    time.sleep(1)
                    prediction = model.predict(img_array)[0][0]
                    if prediction > 0.5:
                        label = "🟢 Healthy Leaf"
                        confidence = prediction
                        healthy_count += 1
                    else:
                        label = "🔴 Diseased Leaf"
                        confidence = 1 - prediction
                        diseased_count += 1

                with cols[idx]:
                    # Display the resized image for consistency
                    original_size = f"{img.size[0]}×{img.size[1]}"
                    processed_size = f"{img_resized.size[0]}×{img_resized.size[1]}"
                    st.image(img_resized, use_container_width=True, 
                            caption=f"📁 {uploaded_file.name} (Original: {original_size} → Processed: {processed_size})")
                    st.markdown(f"""
                        <div class='result-card'>
                            Result: {label}<br>
                            Confidence: {confidence*100:.2f}%
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='summary'>
                    📊 <u>Summary Dashboard</u><br><br>
                    🟢 Healthy Leaves: {healthy_count}<br>
                    🔴 Diseased Leaves: {diseased_count}
                </div>
            """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="upload-box">👈 Please upload one or more guava leaf images to begin analysis.</div>', 
        unsafe_allow_html=True
    )
