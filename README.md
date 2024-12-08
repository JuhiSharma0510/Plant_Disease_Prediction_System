# Plant_Disease_Prediction_System

**View Project** : https://plantdiseasepredictionsystem.streamlit.app/

This repository contains a Plant Disease Prediction System built using Deep Learning. The system identifies diseases in plant leaves based on uploaded images. It is deployed as an interactive web application using Streamlit.This project aims to assist farmers and agricultural experts in diagnosing plant health efficiently.

**Features**<br>
-Deep Learning Model: Pre-trained TensorFlow/Keras model for classifying plant diseases.<br>
-User-Friendly Interface: Upload plant leaf images and get real-time predictions.<br>
-Plant Database: Includes disease information for plants like Apple, Blueberry, Cherry, Corn, and Tomato.<br>

**Project Structure**<br>
├── app.py                    # Main script for running the Streamlit application<br>
├── dataset/                  # Dataset used for training the model<br>
├── models/                   # Pre-trained model (.h5) and class_indices.json file for mapping predictions<br>
├── requirements.txt          # Dependencies for the project<br>
├── README.md                 # Project documentation<br>
├── sample_images/            # Example images for testing<br>


**Prerequisites**<br>
-Python 3.8+<br>
-Install the dependencies: pip install -r requirements.txt<br>

**How to Run**<br>
-Clone the repository: git clone https://github.com/JuhiSharma0510/Plant_Disease_Prediction_System.git<br>
-cd Plant_Disease_Prediction_System<br>
-Start the web app: streamlit run app.py<br>
-Open the app in your browser (default: http://localhost:8501)<br>

**Usage**<br>
-Upload a plant image or input the required details.<br>
-Click the "Predict" button.<br>
-View the prediction and recommendations.<br>

**Technologies Used**<br>
-Programming Language: Python<br>
-Libraries: TensorFlow/Keras, OpenCV, Pandas, NumPy<br>
-Deployment Framework: Streamlit<br>

**Model Details**<br>
The model is trained on a dataset of plant leaf images and can classify various diseases with high accuracy. You can update or retrain the model for additional plants and diseases.

**License**
This project is licensed under the MIT License.

Feel free to contribute or open issues in the repository! 🌱
