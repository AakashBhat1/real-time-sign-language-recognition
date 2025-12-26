This project is a machine learning–based Sign Language Recognition system designed to identify hand gestures and classify them into corresponding sign language symbols. The system follows an end-to-end machine learning pipeline, starting from data collection and preprocessing to model training and real-time prediction using a Python application.

The goal of this project is to help bridge the communication gap between hearing-impaired individuals and others by translating sign language gestures into machine-readable output.

✨ Features

Hand gesture data collection

Data preprocessing and feature extraction

Machine learning model training

Saved trained model for reuse

Real-time sign language prediction

Modular and well-structured Python code

🛠 Tech Stack

Language: Python

Machine Learning: Scikit-learn / TensorFlow (as applicable)

Data Processing: NumPy, Pandas

Model Storage: JSON

Application: Python script (app.py)

📂 Project Structure
├── collectdata.py        # Collects hand gesture data
├── data.py               # Data preprocessing and handling
├── trainmodel.py         # Trains the ML model
├── model.json            # Saved trained model
├── function.py           # Helper and utility functions
├── app.py                # Application for prediction/inference
├── requirements.txt      # Required Python dependencies
└── README.md

⚙️ How the System Works

Hand gesture data is collected using collectdata.py

Raw data is cleaned and processed in data.py

A machine learning model is trained using trainmodel.py

The trained model is saved as model.json

app.py loads the trained model and performs predictions on new inputs

🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Collect data
python collectdata.py

3️⃣ Train the model
python trainmodel.py

4️⃣ Run the application
python app.py
Dataset Notice: This repository does not include the image dataset used for training and testing the model. The project focuses on the implementation of the machine learning workflow and model logic.
