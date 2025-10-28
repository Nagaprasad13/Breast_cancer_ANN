🧬 Breast Cancer Detection using ANN + PCA

A Deep Learning–based Web App that predicts whether a breast tumor is Malignant (Cancerous) or Benign (Non-Cancerous) using Artificial Neural Networks (ANN) and Principal Component Analysis (PCA) for dimensionality reduction.

Built with ❤️ using TensorFlow, Scikit-Learn, and Streamlit.
app=https://breastcancerann-izqjpmkymnsayrya5ym8hb.streamlit.app/
🌟 Features

✅ Cleaned and preprocessed real-world breast cancer dataset
✅ Automatic skewness correction and outlier handling
✅ Scaled features using StandardScaler
✅ Dimensionality reduction with PCA (95% variance)
✅ ANN model built with Keras (ReLU activations + dropout layers)
✅ Web app for live predictions built with Streamlit
✅ Model, scaler, label encoder, PCA saved for reuse (.pkl + .h5)

📁 Project Structure
📦 breast_cancer_ann_pca
├── data.csv                     # Original dataset
├── cleaned_data.csv              # Cleaned version
├── final_training_data_pca.csv   # PCA-transformed training data
├── train_model.py                # Full data cleaning + training pipeline
├── app.py                        # Streamlit web app for predictions
├── model.h5                      # Trained ANN model
├── scaler.pkl                    # Scaler used during training
├── pca.pkl                       # PCA transformer
├── label_encoder.pkl             # Label encoder for target variable
├── feature_names.pkl             # List of training features
└── README.md                     # This file

🧠 Model Architecture
Layer	Type	Activation	Notes
Input	Dense(32)	ReLU	Input = PCA features
Hidden	Dropout(0.2)	—	Regularization
Hidden	Dense(16)	ReLU	—
Hidden	Dropout(0.1)	—	Regularization
Output	Dense(1)	Sigmoid	Binary classification
⚙️ Setup Instructions
1️⃣ Clone the Repo
git clone https://github.com/your-username/breast-cancer-ann-pca.git
cd breast-cancer-ann-pca

2️⃣ Install Dependencies
pip install -r requirements.txt


requirements.txt (example):

pandas
numpy
scikit-learn
tensorflow
streamlit
matplotlib
seaborn
joblib

3️⃣ Train the Model (Optional)

Run this if you want to retrain from scratch:

python train_model.py


This script:

Cleans the dataset

Reduces skewness

Removes outliers

Applies scaling + PCA

Trains the ANN

Saves all artifacts

🚀 Run the Web App

Once the model is trained and saved:

streamlit run app.py


Then open the provided local URL (usually http://localhost:8501/) in your browser.

You’ll see:

A beautiful input form for all features

Model prediction: Malignant (M) or Benign (B)

Confidence score

Friendly visual feedback
