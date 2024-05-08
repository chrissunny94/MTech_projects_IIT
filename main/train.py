import numpy as np
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load pre-trained facial landmark detector from dlib
predictor_path = 'artifacts/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Define model architecture
def build_model():
    # Eye state detection branch
    eye_input = Input(shape=(50, 50, 1))
    eye_conv1 = Conv2D(32, (3, 3), activation='relu')(eye_input)
    eye_pool1 = MaxPooling2D(pool_size=(2, 2))(eye_conv1)
    eye_conv2 = Conv2D(64, (3, 3), activation='relu')(eye_pool1)
    eye_pool2 = MaxPooling2D(pool_size=(2, 2))(eye_conv2)
    eye_flat = Flatten()(eye_pool2)
    eye_dense = Dense(64, activation='relu')(eye_flat)
    eye_output = Dense(1, activation='sigmoid', name='eye_output')(eye_dense)

    # Attention level classification branch
    attention_input = Input(shape=(68,))
    attention_dense1 = Dense(128, activation='relu')(attention_input)
    attention_dense2 = Dense(64, activation='relu')(attention_dense1)
    attention_output = Dense(3, activation='softmax', name='attention_output')(attention_dense2)

    # Combine branches
    combined = concatenate([eye_dense, attention_dense2])
    final_output = Dense(1, activation='sigmoid', name='final_output')(combined)

    model = Model(inputs=[eye_input, attention_input], outputs=[eye_output, attention_output, final_output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Load pre-trained model
model = build_model()
model.load_weights('path_to_pretrained_weights.h5')

# Main loop for real-time inference
camera = cv2.VideoCapture(0)
total = 0

while True:
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_grey, (120, 120))

    dets = detector(frame_resized, 1)

    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Eye state detection
            eye_input = cv2.resize(frame_resized[d.top():d.bottom(), d.left():d.right()], (50, 50))
            eye_input = cv2.cvtColor(eye_input, cv2.COLOR_GRAY2RGB)
            eye_input = np.expand_dims(eye_input, axis=0)
            eye_input = np.expand_dims(eye_input, axis=3)
            eye_state = model.predict(eye_input)

            # Attention level classification
            attention_input = np.reshape(shape, (1, -1))
            attention_level = model.predict(attention_input)

            if ear > 0.25:
                print("Eye aspect ratio:", ear)
                print("Attention level:", attention_level)
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                total += 1
                if total > 20:
                    print("Drowsiness detected")
                    cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, str(total), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
