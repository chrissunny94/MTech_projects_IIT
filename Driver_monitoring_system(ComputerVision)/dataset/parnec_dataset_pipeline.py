import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from imutils import face_utils
import dlib
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
import os

# Download CEW dataset (replace with actual URL)
dataset_url = "https://your_dataset_source.com/cew_dataset.zip"
dataset_path = "cew_dataset.zip"

if not os.path.exists(dataset_path):
    urlretrieve(dataset_url, dataset_path)
    print("Downloaded CEW dataset")
else:
    print("CEW dataset already exists")

# Unzip the dataset using appropriate library (e.g., zipfile)

def calculate_ear(left_eye, right_eye):
    """Calculates the Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(left_eye[1] - left_eye[5])
    B = np.linalg.norm(left_eye[2] - left_eye[4])
    C = np.linalg.norm(right_eye[0] - right_eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def load_data(data_dir, label_file, image_size):
    """Loads eye patch images and labels from a directory"""
    images, labels = [], []
    with open(label_file, 'r') as f:
        for line in f:
            image_path, label = line.strip().split(',')
            image = cv2.imread(os.path.join(data_dir, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (image_size, image_size))
            images.append(image.flatten())
            labels.append(int(label))
    return np.array(images), np.array(labels)

# Initialize dlib's face detector (HOG-based) and shape predictor
path = "artifacts/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

# Load data (replace with actual paths)
image_size = 24
data_dir = "cew_dataset/images"
label_file = "cew_dataset/labels.txt"
X, y = load_data(data_dir, label_file, image_size)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the neural network architecture (for eye closure and sleepiness)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes (eye closed, not closed, sleepiness)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train.reshape(-1, image_size, image_size, 1), y_train, epochs=20, batch_size=32, validation_data=(X_test.reshape(-1, image_size, image_size, 1), y_test))

# Save the model (optional)
model.save('eye_closure_sleepiness_model.h5')


# Function to detect faces, extract features, and predict (replace with your sleepiness logic)
def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
