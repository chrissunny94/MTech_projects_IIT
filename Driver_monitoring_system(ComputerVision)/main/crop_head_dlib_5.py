import cv2
import dlib
import numpy as np
from PIL import Image, ImageOps

#https://linktr.ee/Norod78

MODEL_PATH = "shape_predictor_5_face_landmarks.dat" # You need to download this file from http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
detector = dlib.get_frontal_face_detector() # Initialize dlib's face detector model

def get_face_landmarks(image_path):
    # Load the image
    image = cv2.imread(image_path)
    try:
        image = ImageOps.exif_transpose(image)
    except:
        print("exif problem, not rotating")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize dlib's facial landmarks predictor
    predictor = dlib.shape_predictor("artifacts/shape_predictor_5_face_landmarks.dat")  

    # Detect faces in the image
    faces = detector(gray)

    if len(faces) > 0:
        # Assume the first face is the target, you can modify this based on your requirements
        shape = predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks
    else:
        return None

def calculate_roll_and_yaw(landmarks):
    # Calculate the roll angle using the angle between the eyes
    roll_angle = np.degrees(np.arctan2(landmarks[1, 1] - landmarks[0, 1], landmarks[1, 0] - landmarks[0, 0]))

    # Calculate the yaw angle using the angle between the eyes and the tip of the nose
    yaw_angle = np.degrees(np.arctan2(landmarks[1, 1] - landmarks[2, 1], landmarks[1, 0] - landmarks[2, 0]))

    return roll_angle, yaw_angle

def detect_and_crop_head(input_image, factor=3.0):
    # Get facial landmarks
    landmarks = get_face_landmarks(input_image)

    if landmarks is not None:
        # Calculate the center of the face using the mean of the landmarks
        center_x = int(np.mean(landmarks[:, 0]))
        center_y = int(np.mean(landmarks[:, 1]))

        # Calculate the size of the cropped region
        size = int(max(np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]),
                       np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])) * factor)

        # Calculate the new coordinates for a 1:1 aspect ratio
        x_new = max(0, center_x - size // 2)
        y_new = max(0, center_y - size // 2)

        # Calculate roll and yaw angles
        roll_angle, yaw_angle = calculate_roll_and_yaw(landmarks)

        # Adjust the center coordinates based on the yaw and roll angles
        shift_x = int(size * 0.4 * np.sin(np.radians(yaw_angle)))
        shift_y = int(size * 0.4 * np.sin(np.radians(roll_angle)))

        #print(f'Roll angle: {roll_angle:.2f}, Yaw angle: {yaw_angle:.2f} shift_x: {shift_x}, shift_y: {shift_y}')

        center_x += shift_x
        center_y += shift_y

        # Calculate the new coordinates for a 1:1 aspect ratio
        x_new = max(0, center_x - size // 2)
        y_new = max(0, center_y - size // 2)

        # Read the input image using PIL
        image = Image.open(input_image)

        # Crop the head region with a 1:1 aspect ratio
        cropped_head = np.array(image.crop((x_new, y_new, x_new + size, y_new + size)))

        # Convert the cropped head back to PIL format
        cropped_head_pil = Image.fromarray(cropped_head)

        # Return the cropped head image
        return cropped_head_pil
    else:
        return None

if __name__ == '__main__':
    input_image_path = '/Users/christhaliyath/MTECH/dataset/archive/dms/image/_-2-_mp4-4_jpg.rf.92007ecbe74970aa199e279df7971003.jp'
    output_image_path = 'output.jpg'

    # Detect and crop the head
    cropped_head = detect_and_crop_head(input_image_path, factor=3.0)

    # Save the cropped head image
    cropped_head.save(output_image_path)