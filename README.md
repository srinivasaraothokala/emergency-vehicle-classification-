# emergency-vehicle-classification-


Emergency Vehicle Classification
This repository contains a deep learning model that classifies images of emergency vehicles (e.g., ambulances, police cars, fire trucks) using TensorFlow and Keras. The model is trained to detect emergency vehicles from regular vehicles in image datasets. The repository also includes data preprocessing, model training, and evaluation scripts.

Features
Data Preprocessing: Normalize, reshape, and augment image datasets for better model performance.
Deep Learning Model: A fully connected neural network built using Keras, optimized for binary classification.
Real-Time Detection: Capable of detecting emergency vehicles in images and possibly in video frames.
Easy Integration: The trained model can be integrated into real-time systems for emergency vehicle detection and traffic management.
Technologies Used
Python
TensorFlow/Keras: Deep Learning framework used for building and training the model.
OpenCV: For image processing and augmentations.
Matplotlib/Seaborn: For visualizing datasets and training results.
NumPy: For numerical operations and data handling.
Pandas: For reading and handling CSV data.
Scikit-learn: For splitting the dataset and evaluating the model.
Installation
Clone the repository:

bash
Copy code
git clone <your-repository-url>
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
Dataset:
Upload the dataset to your Google Drive (or local machine) in the following folder structure:

bash
Copy code
/train.csv
/train_images/
The train.csv file should contain image names and their corresponding labels, while the image folder (train_images/) should have the individual images.

Model Training
Run the training script to preprocess data, train the model, and save it:

bash
Copy code
python model_training.py
The model will be trained using the provided data and saved as emergencyModel.h5 in the models/ directory.
The processed datasets are saved as .npz files in the data/ folder to speed up training.
Evaluation
Once trained, you can use the model for predicting emergency vehicles. The trained model (emergencyModel.h5) can be used to classify new images or detect vehicles in video frames.

Example Usage
To classify an image, you can use a separate script or directly load the model:

python
Copy code
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model('models/emergencyModel.h5')

# Load image and preprocess (resize to match input dimensions)
image = cv2.imread('path_to_image.jpg')
image_resized = cv2.resize(image, (224, 224)) / 255.0  # Resize and normalize
image_input = image_resized.reshape(1, 224*224*3)

# Predict emergency vehicle
prediction = model.predict(image_input)
if prediction[0] > 0.5:
    print("Emergency vehicle detected!")
else:
    print("No emergency vehicle detected.")
Directory Structure
bash
Copy code
emergency-vehicle-classification/
│
├── README.md                   # Project documentation
├── model_training.py            # Script to train the model
├── requirements.txt             # Project dependencies
├── data/                        # Directory containing training data
│   ├── train.csv                # CSV with image names and labels
│   ├── temp1.npz                # Processed training data
│   ├── temp2.npz                # Processed validation data
├── models/                      # Folder for saving trained models
│   └── emergencyModel.h5        # Trained model file
└── utils/                       # Utility files for preprocessing and visualization
    └── data_preprocessing.py    # Preprocessing functions
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
TensorFlow and Keras for providing the tools to build the deep learning model.
OpenCV for image preprocessing and manipulation.
Scikit-learn for splitting the dataset and model evaluation tools.
