# Guava Leaf Disease Detection Binary Classifier

## Overview
This project provides an end-to-end solution for detecting diseases in guava leaves using deep learning. It includes:
- A Jupyter notebook for model training and evaluation
- A Streamlit web application for easy image upload and disease prediction

## Features
- **Binary Classification:** Classifies guava leaves as either healthy or diseased.
- **User-Friendly Web UI:** Upload one or more images and get instant predictions with confidence scores.
- **Attractive Design:** Light and dark mode backgrounds for a modern look.
- **Batch Analysis:** Analyze multiple images at once and view a summary dashboard.

## Project Structure
```
GLD_project/
├── GLD_Binary_Classification.ipynb      # Model training and evaluation notebook
├── Green-Guard-Ui/
│   ├── app.py                           # Streamlit web app
│   ├── GLD_Binary_Classification_Final.h5 # Trained Keras model
│   ├── Background_img_2.jpg             # Light mode background
│   ├── Background_dark_img_2.jpg        # Dark mode background
│   └── requirements.txt                 # Python dependencies
└── README.md                            # Project documentation
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rohitgautam1593/Guava_leaf_Disease_detection_Binary_Classifier.git
   cd Guava_leaf_Disease_detection_Binary_Classifier/Green-Guard-Ui
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the web app in your browser (Streamlit will provide a local URL).
- Upload JPG or PNG images of guava leaves.
- Click "Analyze All Uploaded Leaves" to get predictions.
- View the result and summary dashboard.

## Model Training
- The model is trained using the provided Jupyter notebook (`GLD_Binary_Classification.ipynb`).
- The notebook covers data preparation, model architecture, training, and evaluation.
- The trained model is saved as `GLD_Binary_Classification_Final.h5` and used by the web app.

## Credits
- Developed by Rohit Gautam
- Powered by TensorFlow, Keras, and Streamlit

## License
This project is for educational and research purposes.